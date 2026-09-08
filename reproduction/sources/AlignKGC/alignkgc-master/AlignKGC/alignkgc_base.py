import argparse
import copy
import importlib
import random
import sys
from urllib import parse
import numpy
import os
import json
import tempfile
from typing import Dict, List, Set, Tuple
import torch
from torch import cuda, optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import BertTokenizer

from importers.ea_ra_kgc import EaRaKgcData
from mBERT.Param import MbertParams
from AlignKGC.data_loader import data_loader
from AlignKGC import losses, models, evaluate, utils
from AlignKGC.kb import kb

from AlignKGC.BERT_alignments import Basic_Bert_Unit_model, ent2Tokens_gene, \
    ent2bert_input,get_embeddings,cos_sim_mat_generate,batch_topk


def _load_checkpoint(path, map_location="cpu"):
    """Load full Python state on both old and current PyTorch releases."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class AlignKgcBaseTrainer(object):
    """Base class for all trainers."""

    def __init__(self, **kwargs) -> None:
        self.argv = copy.deepcopy(kwargs)
        del self.argv["meta"]

        super(AlignKgcBaseTrainer, self).__init__()

        self.has_cuda = torch.cuda.is_available()
        self.cuda_num = kwargs["cuda_num"]
        if self.has_cuda:
            cuda.set_device(self.cuda_num)
            MbertParams.CUDA_NUM = self.cuda_num
            print("gpu visible:", list(range(cuda.device_count())),
                  "gpu chosen:", self.cuda_num)

        # Using kwargs[argname] means compulsory,
        # using kwargs.get(argname, defval) means optional with default.
        # As a policy reduce defaults to minimum here.
        # Use argparser to inject defaults.

        # Simple configs typically paths and scalar values
        self.meta : EaRaKgcData = kwargs["meta"]
        self.ea_percent : int = kwargs["ea_percent"]
        self.ra_percent : int = kwargs["ra_percent"]
        self.dataset_root : str = \
            self.meta.combined_ea_ra_path(self.ea_percent, self.ra_percent)
        self.max_epochs : int = kwargs["max_epochs"]
        self.eval_only : bool = kwargs["eval_only"]
        self.resume_from_save = kwargs.get("resume_from_save", 0)
        self.resume_checkpoint = kwargs.get("resume_checkpoint")
        if (not self.resume_checkpoint and not self.eval_only and
                kwargs.get("output_root")):
            # Older queue manifests did not pass the new option.  A stable
            # output-root-local fallback lets those already-running queues gain
            # exact resume support without changing their command line.
            self.resume_checkpoint = os.path.join(
                kwargs["output_root"],
                "alignkgc_seed{}_resume.ckpt".format(kwargs["seed"]))
        self.argv["resume_checkpoint"] = self.resume_checkpoint
        self._resume_payload = None
        if self.resume_checkpoint and os.path.isfile(self.resume_checkpoint):
            self._resume_payload = _load_checkpoint(self.resume_checkpoint, "cpu")
            print("found full resume checkpoint", self.resume_checkpoint,
                  "completed_batch", self._resume_payload["completed_batch"])
        output_root = kwargs.get("output_root") or self.dataset_root
        self.model_base : str = \
            os.path.join(output_root, "model_" + self.__class__.__name__)
        if not os.path.isdir(self.model_base):
            print("creating", self.model_base)
            os.mkdir(self.model_base, mode=0o755)

        if self.eval_only:
            self.save_directory : str = None
            self.tflogs_dir : str = None
            self.tbwriter : SummaryWriter = None
        else:
            resumed_directory = (self._resume_payload or {}).get("save_directory")
            if resumed_directory:
                self.save_directory = resumed_directory
                os.makedirs(self.save_directory, exist_ok=True)
                print("resuming model in", self.save_directory)
            else:
                self.save_directory = tempfile.mkdtemp(dir=self.model_base)
                print("saving model to", self.save_directory)
                os.chmod(self.save_directory, 0o755)
                commit_path = os.path.join(self.save_directory, "commit.json")
                assert not os.path.isfile(commit_path), \
                    "{} exists, quitting".format(commit_path)
            self.tflogs_dir = self.save_directory
            purge_step = None
            if self._resume_payload:
                purge_step = self._resume_payload["completed_batch"] + 1
            self.tbwriter = SummaryWriter(log_dir=self.save_directory,
                                          purge_step=purge_step)
            out_err_path = os.path.join(self.save_directory, "log.txt")
            self.saved_stdout = sys.stdout
            self.saved_stderr = sys.stderr
            sys.stdout = sys.stderr = utils.tee(
                out_err_path, append=bool(self._resume_payload))

        self.verbose : int = kwargs["verbose"]
        self.batch_size : int = kwargs["batch_size"]
        self.negative_sample_count : int = kwargs["negative_sample_count"]
        self.eval_batch_size : int = kwargs["eval_batch_size"]
        self.minimum_eval_batch_size : int = kwargs["minimum_eval_batch_size"]
        self.eval_every_x_mini_batches : int = \
            kwargs["eval_every_x_mini_batches"]
        self.tensorboard_log_every : int = kwargs["tensorboard_log_every"]
        self.gradient_clip : float = kwargs["gradient_clip"]
        self.regloss_coeff : float = kwargs["regloss_coeff"]
        self.ealoss_coeff : float = kwargs["ealoss_coeff"]
        self.raloss_coeff : float = kwargs["raloss_coeff"]
        if kwargs["hooks"]:
            self.hooks = json.loads(kwargs["hooks"])
        else:
            self.hooks = list()
        self.loss = getattr(losses, kwargs["loss"])()

        # Track validation perf
        self.best_mrr_on_valid : Dict = None

        # KGC train dev test folds and dependent members
        self.dltrain : data_loader = None
        self.dlvalid : data_loader = None
        self.dltestmap : Dict[str, Tuple[str, data_loader]] = None
        self.scoring_function = None
        self.optim : optim = None
        self.scheduler : lr_scheduler.ReduceLROnPlateau = None
        self.filtmap : Dict[str, List] = None
        self.init_folds(**kwargs)

        # These may be best to init in subclasses.
        # self.ent_aligns : dict[int,int] = kwargs["ent_aligns"]
        # """ dict(g_ent_id, g_ent_id) """
        # self.rel_aligns = kwargs["rel_aligns"]
        # self.rel_implies = kwargs["rel_implies"]


    def init_folds(self, **kwargs):
        first_zero_val = (kwargs["oov_entity"] != None)

        ktrain = kb(os.path.join(self.dataset_root, 'train.txt'))
        # Reproduction-only smoke-test limiter.  The default (0) preserves the
        # upstream behaviour and uses every fact.  Apply the slice only after
        # ``kb`` has built the complete entity/relation maps so that tensor
        # shapes remain identical to a full-data run.
        max_samples = kwargs.get("MAX_SAM", 0)
        if max_samples and max_samples > 0:
            ktrain.facts = ktrain.facts[:max_samples]
        if kwargs["oov_entity"]:
            if not "<OOV>" in ktrain.entity_map.keys():
                ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
                ktrain.nonoov_entity_count = ktrain.entity_map["<OOV>"]+1

        self.dltrain : data_loader = \
            data_loader(ktrain, self.has_cuda, loss=self.loss,
                        flag_add_reverse=kwargs["inverse"],
                        first_zero=first_zero_val)
        kvalid = kb(os.path.join(self.dataset_root,'valid.txt'),
                       em=ktrain.entity_map, rm=ktrain.relation_map,
                       add_unknowns=not kwargs["oov_entity"],
                       nonoov_entity_count=ktrain.nonoov_entity_count)
        if max_samples and max_samples > 0:
            kvalid.facts = kvalid.facts[:max_samples]
        self.dlvalid : data_loader = \
            data_loader(kvalid, self.has_cuda, loss=self.loss,
                        first_zero=first_zero_val)

        ktestmap = dict()
        """ key = langname below; val = (lang, kb) """
        for lang in self.meta.langs:
            for langname in ["test_" + lang + ".txt",
                            lang + "_f_test.txt",
                            lang + "_o_test.txt"]:
                langkpath = os.path.join(self.dataset_root, langname)
                print("loading", langname)
                langkb = kb(langkpath, em=ktrain.entity_map,
                            rm=ktrain.relation_map,
                            add_unknowns=not kwargs["oov_entity"],
                            nonoov_entity_count =
                            ktrain.nonoov_entity_count)
                if max_samples and max_samples > 0:
                    langkb.facts = langkb.facts[:max_samples]
                ktestmap[langname] = (lang, langkb)
        self.dltestmap : Dict[str, Tuple[str, kb]] = dict()
        """ key = langname; val = (lang, dlkb) """
        for langname, (lang, langkb) in ktestmap.items():
            dlkb = data_loader(langkb, self.has_cuda, loss=self.loss,
                               first_zero=first_zero_val)
            self.dltestmap[langname] = (lang, dlkb)

        self.filtmap = dict()
        """ key = lang; val = filt """
        if kwargs["filter"]:
            for lang in self.meta.langs:
                filtname = "filters_" + lang + ".txt"
                filtpath = os.path.join(self.dataset_root, filtname)
                print("loading", filtname)
                langfilt = utils.get_filter(filtpath, em=ktrain.entity_map,
                                            rm=ktrain.relation_map,
                                            add_unknowns=
                                            not kwargs["oov_entity"],
                                            nonoov_entity_count =
                                            ktrain.nonoov_entity_count)
                self.filtmap[lang] = langfilt

        model_arguments = json.loads(kwargs["model_arguments"])
        model_arguments['entity_count'] = len(ktrain.entity_map)
        if kwargs["regularizer"]:
            print("Using reg ", kwargs["regularizer"])
            model_arguments['reg'] = kwargs["regularizer"]
        if kwargs["inverse"]:
            model_arguments['relation_count'] = len(ktrain.relation_map)*2
            model_arguments['flag_add_reverse'] = kwargs["inverse"]
            model_arguments['flag_avg_scores'] = kwargs["avg_scores"]
        else:
            model_arguments['relation_count'] = len(ktrain.relation_map)
        model_arguments['batch_norm'] = kwargs["batch_norm"]
        print("model_arguments", model_arguments)

        self.scoring_function = getattr(models, kwargs["model"])(**model_arguments)
        if self.has_cuda:
            self.scoring_function = self.scoring_function.cuda()
        self.regularizer = self.scoring_function.regularizer
        try:
            self.flag_add_reverse = self.scoring_function.flag_add_reverse
        except:
            self.flag_add_reverse = 0

        self.optim = getattr(torch.optim, kwargs["optimizer"])\
            (self.scoring_function.parameters(), lr=kwargs["learning_rate"])
        # ``verbose`` was removed from ReduceLROnPlateau in recent PyTorch.
        # It only controlled console logging, so omitting it preserves the
        # optimization schedule while supporting both old and current builds.
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
            'max', patience = 2)
        if not self.eval_batch_size:
            self.eval_batch_size = max(50, self.batch_size*2*
                self.negative_sample_count//len(ktrain.entity_map))
        if self.minimum_eval_batch_size:
            self.eval_batch_size = max(self.eval_batch_size,
                                       self.minimum_eval_batch_size)

    def get_bert_entity_alignments(self, entmap=None):
        BERT_MODEL_PATH = self.argv["mbert_path"]
        BERT_MODEL_FILE = os.path.join(BERT_MODEL_PATH, "combined_mBERT.p")
        TOKENIZER_PATH = "bert-base-multilingual-cased"
        dbp5lentlist_path = os.path.join(self.argv["dbp5l"], "entity_lists/")
        entmap_path = os.path.join(self.dataset_root,'mapping.txt')
        bert_dir_path = BERT_MODEL_PATH
        model_pkl_path = BERT_MODEL_FILE
        tokenizer_path = TOKENIZER_PATH

        print("Getting Alignment from mBERT.................\n",
            "entmap_path", entmap_path, "\n", "bert_dir_path", bert_dir_path, "\n",
            "model_pkl_path", model_pkl_path, "\n", "tokenizer_path", tokenizer_path, "\n",
            "dbp5lentlist_path", dbp5lentlist_path)

        mapping = {}
        with open(entmap_path) as f:
            lines=f.readlines()
            for line in lines:
                a=line.split()
                mapping[(a[2],int(a[1]))]=a[0]

        Tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print("tokenizer loaded from", tokenizer_path)
        Model = Basic_Bert_Unit_model(768,300,bert_dir_path).cuda()
        print("model loaded from", bert_dir_path)
        Model.load_state_dict(torch.load(open(model_pkl_path,"rb")))
        print("state_dict loaded from", model_pkl_path)

        entities, entids = self.meta.get_entities()
        ent2tokenids = ent2Tokens_gene(Tokenizer,entids)
        ent2data = ent2bert_input(entids,Tokenizer,ent2tokenids)

        emb: Dict[str, List] = dict()
        for langx in self.meta.langs:
            emb[langx] = get_embeddings(Model, entities[langx], ent2data)

        alignment : Dict[Tuple[str], List] = dict()
        for (lang1, lang2) in self.meta.lang_pairs:
            print(lang1, lang2)
            assert lang1 != lang2
            res_mat12 = cos_sim_mat_generate(emb[lang1],emb[lang2])
            score12, top_index12 = batch_topk(res_mat12,topn=1,largest=True)
            # nearest lang2 nbr for each ent in lang1
            if (lang1,lang2) not in alignment:
                alignment[(lang1,lang2)]=list(zip(score12.view(-1).tolist(),
                                                top_index12.view(-1).tolist()))
            res_mat21 = cos_sim_mat_generate(emb[lang2],emb[lang1])
            score21, top_index21 = batch_topk(res_mat21,topn=1,largest=True)
            # nearest lang1 nbr for each ent in lang2
            if (lang2,lang1) not in alignment:
                alignment[(lang2,lang1)]=list(zip(score21.view(-1).tolist(),
                                                top_index21.view(-1).tolist()))

        predicted_alignments = list()
        for (lang1, lang2) in self.meta.lang_pairs:
            print(lang1, lang2)
            assert lang1 != lang2
            align12 = alignment[(lang1, lang2)]
            align21 = alignment[(lang2, lang1)]
            for index1, (score12, index12) in enumerate(align12):
                score21, index21 = align21[index12]
                if index21==index1 and (lang1,index1) in mapping and \
                        (lang2,index12) in mapping and \
                        mapping[(lang1,index1)] != mapping[(lang2,index12)]:
                    x=mapping[(lang1,index1)]
                    y=mapping[(lang2,index12)]
                    if x in entmap and y in entmap:
                        predicted_alignments.append((entmap[x], score12, entmap[y]))

        return predicted_alignments

    def find_best_dev_kgc_model(self) -> str :
        """Search for model with best dev perf under a directory."""
        max_dev_perf = 0
        argmax_dev_path = None
        argmax_dev_perf = None
        for runname in os.listdir(self.model_base):
            runpath = os.path.join(self.model_base, runname)
            perfpath = os.path.join(runpath, "commit.json")
            if not os.path.isfile(perfpath):
                continue
            print("\tscan dev perf", runname)
            with open(perfpath, "rb") as perf_file:
                perfs = json.load(perf_file)
                if max_dev_perf < perfs["valid_score"]["m"]["mrr"]:
                    max_dev_perf = perfs["valid_score"]["m"]["mrr"]
                    argmax_dev_perf = perfs
                    argmax_dev_path = runpath
        if argmax_dev_perf:
            print("best dev perf", argmax_dev_path,
                  "macro_e2_mrr", argmax_dev_perf["valid_score"]["e2"]["mrr"],
                  "macro_m_mrr", argmax_dev_perf["valid_score"]["m"]["mrr"])
        return argmax_dev_path

    def do_eval_kgc_only(self):
        """Search for model with best dev perf under a directory, load that
        model, evaluate on test fold."""

        assert self.eval_only, "Flag eval_only not set"
        best_model_dir = self.find_best_dev_kgc_model()
        best_model_path = os.path.join(best_model_dir, "best_valid_model.pt")
        best_model = _load_checkpoint(
            best_model_path, map_location="cuda:{}".format(self.cuda_num))
        self.scoring_function.load_state_dict(best_model["model_weights"])
        test_scores_per_lang = dict()
        sum_e2_mrr = 0.
        save_path = os.path.join(best_model_dir, "save_eval.csv")
        with open(save_path, "w") as save_file:
            save_eval = utils.SaveEval(save_file)
            for langname, (langx, langkb) in self.dltestmap.items():
                test_score = evaluate.evaluate(langname, self.ranker,
                    langkb.kb, self.eval_batch_size,
                    verbose=self.verbose, hooks=self.hooks,
                    filt = self.filtmap[langx],
                    save_eval=save_eval)
                test_scores_per_lang[langname] = test_score
                sum_e2_mrr += test_score["e2"]["mrr"]
        print("macro e2 mrr", sum_e2_mrr / len(self.dltestmap))

    def start(self, steps: int=None, batch_count: int=None, mb_start: int=None):
        """Entry point for training."""
        assert not self.eval_only, "Cannot train in eval_only mode"
        if steps is None:
            steps = int(self.max_epochs * self.dltrain.kb.facts.shape[0] /
                        self.batch_size)
        print("steps=%d, eval_batch_size=%d" % (steps, self.eval_batch_size))
        if batch_count is None:
            batch_count = [self.eval_every_x_mini_batches//20, 20]
        print("batch_count", batch_count)
        if mb_start is None:
            if self._resume_payload:
                mb_start = self.load_resume_checkpoint()
            elif self.resume_from_save:
                mb_start = self.load_state(self.resume_from_save)
            else:
                mb_start = 0
        print("mb_start", mb_start)
        losses = []
        count = 0
        print("Starting training")
        for batchx in range(mb_start, steps):
            various_losses = self.step(batchx)
            losses.append(various_losses["total"])
            if self.tensorboard_log_every > 0 and \
                    batchx % self.tensorboard_log_every == 0:
                self.tbwriter.add_scalars("loss", various_losses, batchx)
            if len(losses) >= batch_count[0]:
                count += 1
                losses = []
                if count == batch_count[1]:
                    self.scoring_function.eval()
                    valid_score = evaluate.evaluate("valid", self.ranker,
                        self.dlvalid.kb, self.eval_batch_size, 
                        verbose=self.verbose, hooks=self.hooks)
                    self.tbwriter.add_scalar("valid_e2_mrr",
                        valid_score["e2"]["mrr"], batchx)
                    self.tbwriter.flush()
                    test_scores_per_lang = dict()
                    for langname, (langx, langkb) in self.dltestmap.items():
                        test_score = evaluate.evaluate(langname, self.ranker,
                            langkb.kb, self.eval_batch_size, 
                            verbose=self.verbose, hooks=self.hooks,
                            filt = self.filtmap[langx])
                        test_scores_per_lang[langname] = test_score
                    self.scoring_function.train()
                    self.scheduler.step(valid_score['m']['mrr'])
                    #Scheduler to manage learning rate added
                    count = 0
                    self.save_state(batchx, valid_score, test_scores_per_lang)
                    self.save_resume_checkpoint(batchx)
                    self.tbwriter.flush()
        self.terminate()

    def terminate(self):
        commit_dict = copy.deepcopy(self.argv)
        commit_dict.update(self.best_mrr_on_valid)
        commit_path = os.path.join(self.save_directory, "commit.json")
        with open(commit_path, "w") as commit_file:
            json.dump(commit_dict, commit_file)
        print("wrote", commit_path)
        self.tbwriter.flush()
        self.tbwriter.close()
        sys.stderr = self.saved_stderr
        sys.stdout = self.saved_stdout

    def save_state(self, mini_batches, valid_score, test_scores):
        state = dict()
        state['mini_batches'] = mini_batches
        state['epoch'] = mini_batches*self.batch_size/self.dltrain.kb.facts.shape[0]
        state['model_name'] = type(self.scoring_function).__name__
        state['model_weights'] = self.scoring_function.state_dict()
        state['optimizer_state'] = self.optim.state_dict()
        state['optimizer_name'] = type(self.optim).__name__
        state['entity_map'] = self.dltrain.kb.entity_map
        state['reverse_entity_map'] = self.dltrain.kb.reverse_entity_map
        state['relation_map'] = self.dltrain.kb.relation_map
        state['reverse_relation_map'] = self.dltrain.kb.reverse_relation_map
        state['nonoov_entity_count'] = self.dltrain.kb.nonoov_entity_count
        state["valid_score"] = valid_score
        state['test_scores'] = test_scores

        if not self.best_mrr_on_valid or \
                state['valid_score']['m']['mrr'] >= \
                    self.best_mrr_on_valid["valid_score"]["m"]["mrr"]:
            # print("_ARGV_", str(self.argv))
            # print("_BEST_MODEL_ {}".format(state["valid_score"]))
            best_name = os.path.join(self.save_directory, "best_valid_model.pt")
            self.best_mrr_on_valid = {
                "valid_score" : copy.deepcopy(valid_score),
                "test_scores" : copy.deepcopy(test_scores)
            }
            temporary = best_name + ".tmp.{}".format(os.getpid())
            torch.save(state, temporary)
            os.replace(temporary, best_name)
            print("saved state to", best_name)

    def load_state(self, state_file):
        state = _load_checkpoint(
            state_file, map_location="cuda:{}".format(self.cuda_num))
        if state['model_name'] != type(self.scoring_function).__name__:
            utils.colored_print('yellow', 'model name in saved file %s is different from the name of current model %s' %
                                (state['model_name'], type(self.scoring_function).__name__))
        self.scoring_function.load_state_dict(state['model_weights'])
        if state['optimizer_name'] != type(self.optim).__name__:
            utils.colored_print('yellow', ('optimizer name in saved file %s is different from the name of current '+
                                          'optimizer %s') %
                                (state['optimizer_name'], type(self.optim).__name__))
        self.optim.load_state_dict(state['optimizer_state'])
        return state['mini_batches'] + 1

    def save_resume_checkpoint(self, completed_batch):
        """Atomically save all state needed to continue at the next batch."""
        if not self.resume_checkpoint:
            return
        equiv_rel = {}
        for lid, matches in getattr(self, "equiv_rel", {}).items():
            equiv_rel[lid] = {
                rel: (score.detach().cpu(), rel2)
                for rel, (score, rel2) in matches.items()
            }
        state = {
            "schema_version": 1,
            "completed_batch": completed_batch,
            "save_directory": self.save_directory,
            "model_name": type(self.scoring_function).__name__,
            "model_weights": self.scoring_function.state_dict(),
            "optimizer_name": type(self.optim).__name__,
            "optimizer_state": self.optim.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_mrr_on_valid": copy.deepcopy(self.best_mrr_on_valid),
            "equiv_rel": equiv_rel,
            "b": getattr(self, "b", None).detach().cpu()
                 if hasattr(self, "b") else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": numpy.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [state.cpu() for state in
                                torch.cuda.get_rng_state_all()]
                               if torch.cuda.is_available() else [],
        }
        parent = os.path.dirname(os.path.abspath(self.resume_checkpoint))
        os.makedirs(parent, exist_ok=True)
        temporary = self.resume_checkpoint + ".tmp.{}".format(os.getpid())
        torch.save(state, temporary)
        os.replace(temporary, self.resume_checkpoint)
        print("saved full resume checkpoint to", self.resume_checkpoint,
              "completed_batch", completed_batch)

    def load_resume_checkpoint(self):
        """Restore a full-state checkpoint and return the next batch index."""
        state = self._resume_payload
        if state is None:
            state = _load_checkpoint(self.resume_checkpoint, "cpu")
        if state.get("schema_version") != 1:
            raise ValueError("unsupported AlignKGC checkpoint schema")
        if state["model_name"] != type(self.scoring_function).__name__:
            raise ValueError("checkpoint model does not match current model")
        if state["optimizer_name"] != type(self.optim).__name__:
            raise ValueError("checkpoint optimizer does not match current optimizer")
        self.scoring_function.load_state_dict(state["model_weights"])
        self.optim.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.best_mrr_on_valid = copy.deepcopy(state["best_mrr_on_valid"])
        device = next(self.scoring_function.parameters()).device
        self.equiv_rel = {
            lid: {rel: (score.to(device), rel2)
                  for rel, (score, rel2) in matches.items()}
            for lid, matches in state["equiv_rel"].items()
        }
        if state.get("b") is not None and hasattr(self, "b"):
            self.b.data.copy_(state["b"].to(self.b.device))
        random.setstate(state["python_rng_state"])
        numpy.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(
                [rng.cpu() for rng in state["cuda_rng_states"]])
        next_batch = state["completed_batch"] + 1
        print("restored full resume checkpoint", self.resume_checkpoint,
              "next_batch", next_batch)
        return next_batch

    def get_lrid_to_soset_map(self):
        """Prepare map from lrid to SO-pairs(rel). INCOMPLETE."""
        lrid_to_soset : Dict[int, List[Tuple[int,int]]] = dict()
        for lang in self.meta.langs:
            lid = self.meta.lang_to_lid(lang)
            lang_kgc_train_path = os.path.join(self.meta.dir,
                                               "kgs/" + lang + "-train.tsv")
            for (sid, rid, oid) in self.meta.read_tsv_int_path(lang_kgc_train_path):
                lrid = self.meta.lang_rel_do_prefix(lid, rid)
                assert int == type(lrid)
                gsid = self.dltrain.kb.entity_map[str(sid)]
                assert int == type(gsid)
                goid = self.dltrain.kb.entity_map[str(oid)]
                assert int == type(goid)
                if lrid not in lrid_to_soset:
                    lrid_to_soset[lrid] = list()
                lrid_to_soset[lrid].append((gsid, goid))
        print("lrid_to_soset", len(lrid_to_soset))
        return lrid_to_soset

    def ent_alignment_loss(self, align, E_re, E_im):
        loss=[]
        for ent1,sc,ent2 in align:
            p1=torch.tensor([ent1]).cuda()
            p2=torch.tensor([ent2]).cuda()
            loss.append(sc*(E_re(p1)-E_re(p2)).abs().mean()
                        +sc*(E_im(p1)-E_im(p2)).abs().mean())
        return sum(loss)/len(loss)

    def ra_loss_hard(self, meta: EaRaKgcData,
                     prid_to_sos: Dict[int, List[Tuple[int,int]]],
                     dojaccard: bool):
        """Replacement for get_rel_align_imply and rel_alignment_loss.
        INCOMPLETE."""
        ra_loss_ans : float = 0
        rids: List[int] = list()
        print("opening", meta.ra_path(self.ra_percent))
        for [rid] in meta.read_tsv_int_path(meta.ra_path(self.ra_percent)):
            rids.append(rid)  # no prefix
        num_tried, num_found = 0, 0
        for rid in rids:
            for lang1 in meta.langs:
                ll1 = meta.lang_to_lid(lang1)
                for lang2 in meta.langs:
                    if lang1 >= lang2:
                        continue
                    ll2 = meta.lang_to_lid(lang2)
                    ll1rid = meta.lang_rel_do_prefix(ll1, rid)
                    ll2rid = meta.lang_rel_do_prefix(ll2, rid)
                    num_tried += 1
                    if ll1rid in prid_to_sos and ll2rid in prid_to_sos:
                        num_found += 1
        print("rids", len(rids), "found", num_found, "of", num_tried)
        return ra_loss_ans


def get_base_argparser():
    """Prepares and returns an argparser with command line arguments shared
    across all AlignKGC variations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbp5l", required=True, help="/path/to/DBP-5L/")
    parser.add_argument("--mbert_path", help="/path/to/data/bert_cased/")
    parser.add_argument("--eval_only", default=False, action='store_true')
    parser.add_argument('--loss', default="crossentropy_loss_AllNeg_subsample",
                        help="loss function name as in losses.py")
    parser.add_argument('--model', default="complex", required=False,
                        help="model name as in models.py")
    parser.add_argument('--model_arguments', default=
                        '{"embedding_dim":180, "batch_norm":1, "unit_reg":0}',
                        help="model arguments as in __init__ of "
                        "model (Excluding entity and relation count) "
                        "This is a json string", required=False)
    parser.add_argument('--optimizer', default='Adagrad')
    parser.add_argument('--learning_rate', type=float, default=0.8)
    parser.add_argument('--regularizer', default=2.0, type=float,
                        choices=[2.0, 3.0], help="regularizer norm")
    parser.add_argument('--regloss_coeff', type=float, default=0.02)
    parser.add_argument("--ealoss_coeff", type=float, default=50.0)
    parser.add_argument("--raloss_coeff", type=float, default=5.)
    parser.add_argument("--ea_percent", type=int, default=20)
    parser.add_argument("--ra_percent", type=int, default=20)
    parser.add_argument('--gradient_clip', type=float)
    parser.add_argument('--max_epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--eval_every_x_mini_batches', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=0)
    parser.add_argument('--minimum_eval_batch_size', type=int, default=256,
                        help='semantics-neutral lower bound for eval chunks; 0 disables')
    parser.add_argument('--tensorboard_log_every', type=int, default=100,
                        help='write training losses every N batches; 0 disables')
    parser.add_argument('--negative_sample_count', type=int, default=2000)
    parser.add_argument('--MAX_SAM', type=int, default=0,
                        help='smoke-test fact cap per split; 0 keeps full data')
    parser.add_argument('--resume_from_save', default=0,
                        help='legacy best-model checkpoint path')
    parser.add_argument('--resume_checkpoint',
                        help='atomic full-state checkpoint path')
    parser.add_argument('--oov_entity', type=int, default=1)
    parser.add_argument('-q', '--verbose', type=int, default=0)
    parser.add_argument('-z', '--debug', type=int, default=0)
    parser.add_argument('-k', '--hooks', default="[]")
    parser.add_argument('-bn', '--batch_norm', type=int, default=0)
    parser.add_argument('-msg', '--message', required=False)
    parser.add_argument('-f', '--filter', type=int, default=1)
    parser.add_argument('-inv', '--inverse', type=int, default=0)
    parser.add_argument('-avg', '--avg_scores', default=0)
    parser.add_argument("--seed", type=int, default=41,
                        help="seed for numpy and torch random numbers")
    parser.add_argument("--multiseed", type=int, default=1,
                        help="If >1, use seed to generate multiseed seeds "
                        "and run multiple times saving to different model dirs")
    parser.add_argument("--cuda_num", type=int, default=0,
                        help="Ordinal number of cuda device 0/1/2 etc.")
    parser.add_argument("--output_root",
                        help="optional model-output root; dataset remains read-only")
    return parser


def main(av: argparse.Namespace, modname: str, classname: str):
    """Run trainer once or multiple times with random seeds.
    AlignKgcBaseTrainer and subclasses do not use av.seed or 
    av.multiseed and can be constructed before calling this method."""
    if av.multiseed > 1:
        seeds = numpy.random.randint(10000, size=av.multiseed)
    else:
        seeds = [av.seed]
    for seed in seeds:
        av.seed = int(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        module_ = importlib.import_module(modname)
        class_ = getattr(module_, classname)
        meta = EaRaKgcData(av.dbp5l)
        trainer: AlignKgcBaseTrainer = class_(meta=meta, **(av.__dict__))
        if trainer.eval_only:
            trainer.do_eval_kgc_only()
        else:
            trainer.start()


if __name__ == "__main__":
    parser = get_base_argparser()
    parser.add_argument("--trainer_module", required=True)
    parser.add_argument("--trainer_class", required=True)
    av = parser.parse_args()
    main(av, av.trainer_module, av.trainer_class)
