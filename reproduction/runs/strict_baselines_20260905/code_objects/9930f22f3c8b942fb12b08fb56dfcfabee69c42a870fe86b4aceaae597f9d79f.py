import os
import tqdm
from typing import Dict, List, TextIO, Tuple


class EaRaKgcData(object):
    """Loads up meta data from a base directory holding a data set,
    like DBP-5L, for EA, RA and KGC.
    Some naming conventions:
    * rid = global relation id without any prefix
    * urid = relation id with union language prefix
    * lrid = relation id with its own language prefix
    * prid = relation id with some prefix, u or l
    * geid, gsid, goid = global entity, subject, object id
    """

    def __init__(self, earakgc_dir):
        """
        Collects all lang codes in the data set by looking for
        alignment files.  Cannot look for entity lists because DBP5L
        has 'de' without alignments :-(
        :param earakgc_dir:
        """
        self.dir = earakgc_dir
        ea_dir = os.path.join(self.dir, "seed_alignment")
        self.langs : set[str] = set()
        self.lang_pairs : set[tuple[str]] = set()
        for ea_file in [f for f in os.listdir(ea_dir)
                        if os.path.isfile(os.path.join(ea_dir, f))]:
            lang_pair = os.path.splitext(ea_file)[0].split('-')
            self.lang_pairs.add(tuple(lang_pair))
            for lang in lang_pair:
                self.langs.add(lang)
        # canonical orders of lang and lang pair are important
        self.langs = sorted(list(self.langs))
        self.lang_pairs = sorted(list(self.lang_pairs))
        # cannot afford more than 8 languages otherwise there may be
        # ID collisions after prefixing with lang id
        assert len(self.langs) < 9, "Cannot support >8 languages"
        print(self.__dict__)

    def ea_path(self, langA: str, langB: str, ea_percent: int):
        """Sampled alignment path."""
        return os.path.join(self.dir, "seed_alignment_" +
                            str(ea_percent) + "/" +
                            langA + "-" + langB + ".tsv")

    def ra_path(self, ra_percent: int):
        """In DBP-5L, relation IDs are global to start with, and therefore
        already aligned.  We `unalign' a fraction of them and rename them
        for each language.  The fraction this remains aligned with global
        IDs, is in the file given by the returned path.
        Other data sets like WikiData may need a different treatment/API."""
        return os.path.join(self.dir, "relat_" + str(ra_percent) + ".txt")

    def combined_ea_ra_path(self, ea_percent: int, ra_percent: int) -> str:
        return os.path.join(self.dir, "combined/Combined_" +
                            str(ea_percent) + "_" + str(ra_percent))

    def lang_to_lid(self, lang : str) -> int :
        """Given 2-char lang, return an int between 1 and
        len(self.langs) inclusive, based on lexicographic
        order.  We cannot use 0 because we prefix relation IDs
        with this number.  Fails if not found."""
        assert len(self.langs) < 9, "Cannot support >8 languages"
        return 1 + self.langs.index(lang)

    def lid_to_lang(self, lid: int) -> str:
        """Input is between 1 and maxlid.
        Output is 2-letter lang as str"""
        assert len(self.langs) < 9, "Cannot support >8 languages"
        return self.langs[lid - 1]

    def maxlid_plus_one(self) -> int:
        assert len(self.langs) < 9, "Cannot support >8 languages"
        return 1 + len(self.langs)

    def lids(self):
        """return all valid integer language ids for this data set"""
        return range(1, self.maxlid_plus_one())

    def lang_rel_do_prefix(self, prefix: int, rid: int) -> int:
        """No separator after prefix so we cannot support more
        than one prefix (lang) digit."""
        assert int == type(prefix), type(prefix)
        assert int == type(rid), type(rid)
        assert 0 < prefix and prefix < 9, "Cannot support >8 languages"
        return int(str(prefix) + str(rid))

    def lang_rel_un_prefix(self, prid: int) -> int:
        """Remove first digit as lang prefix.  No separator!"""
        assert len(self.langs) < 9, "Cannot support >8 languages"
        assert int == type(prid)
        return int(str(prid)[1:])

    def dbpedia_prefix_len(self, lang : str) -> int:
        """Should return 28 for lang=en and 31 otherwise.
        TODO Peek into entity files and set prefixes adaptively """
        if lang == "en":
            return len("http://dbpedia.org/resource/")
        else:
            return len("http://fr.dbpedia.org/resource/")

    def get_entities(self) -> Tuple[Dict[str, List[str]], List[str]]:
        entities : Dict[str, List[str]] = dict()
        """ key = lang, val = listof ent str """

        ents_dir= os.path.join(self.dir, "entity_lists")
        for langx in self.langs:
            entities[langx] = list()
            plenx = self.dbpedia_prefix_len(langx)  # strip dbpedia URL prefix
            with open(os.path.join(ents_dir, langx + ".tsv")) as ents_file:
                for row in tqdm.tqdm(ents_file, desc=langx):
                    entities[langx].append(row[plenx:-1])

        entids : list[str] = list()
        """concat of all ent str in all langs in lex order;
        order may matter to BERT"""
        for langx in self.langs:
            entids.extend(entities[langx])

        return entities, entids

    def read_tsv_int_path(self, apath: str):
        """Opens apath, reads line by line, returns list of ints per line"""
        with open(apath) as afile:
            for line in afile:
                yield [int(x) for x in line.split()]

    def read_tsv_int_file(self, afile: TextIO):
        """File has already been opened;
        reads line by line, returns list of ints per line"""
        for line in afile:
            yield [int(x) for x in line.split()]
