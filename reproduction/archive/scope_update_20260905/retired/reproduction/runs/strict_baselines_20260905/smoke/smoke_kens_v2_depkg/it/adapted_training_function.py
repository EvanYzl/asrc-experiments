def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    set_params(args)

    target_lang = param.lang
    src_langs = ['de', 'es', 'fr', 'it', 'jp', 'uk']
    src_langs.remove(target_lang)

    target_lang = param.lang

    # load data
    data_dir = 'G:\\zhishitupui\\data\\raw\\dmkgc\\datasetdepkg\\kg'  # where you put kg data
    seed_dir = 'G:\\zhishitupui\\data\\raw\\dmkgc\\datasetdepkg\\seed_alignlinks'  # where you put seed align links data
    model_dir = 'G:\\zhishitupui\\reproduction\\runs\\strict_baselines_20260905\\smoke\\smoke_kens_v2_depkg\\it\\model'  # output
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # logging
    set_logger(param, model_dir)  # set logger
    logging.info('Knowledge model: %s'%(param.knowledge))
    logging.info('target language: %s'%(param.lang))

    # hyper-parameters
    logging.info(f'dim: {param.dim}')
    logging.info(f'lr: {param.lr}')


    # target (sparse) kg
    kg0 = load_target_kg(data_dir, target_lang, testfile_suffix='-val.tsv')  # target kg. KnowledgeGraph object
    # supporter kgs
    supporter_kgs = load_support_kgs(data_dir, seed_dir, target_lang, src_langs)  # list[KnowledgeGraph]
    # KnowledgeGraph.__init__ already adds supporter validation facts once.
    # Do not append them a second time and silently double their training weight.

    # seed alignment links
    seed_alignlinks = load_all_to_all_seed_align_links(seed_dir)  # {(lang1, lang2): 2-col np.array}

    if args.MAX_SAM < 10000000000:
        for kg in [kg0] + supporter_kgs:
            for split in ('train', 'test'):
                for field in ('h', 'r', 't', 'y'):
                    attr = f'{field}_{split}'
                    if hasattr(kg, attr):
                        setattr(kg, attr, getattr(kg, attr)[:args.MAX_SAM])
        seed_alignlinks = {
            pair: links[:args.MAX_SAM]
            for pair, links in seed_alignlinks.items()
        }

    all_kgs = [kg0] + supporter_kgs

    # build alignment model (all-to-all)
    for kg in all_kgs:
        kg.build_alignment_models(all_kgs)  # kg.align_models_of_all {lang: align_model}

    # create validator
    validator = MultiModelTester(kg0, supporter_kgs)

    print('model initialization done')

    for i in range(param.round):
        # train alignment model
        for kg in all_kgs:
            # align it with everything else
            for other_lang, align_model in kg.align_models_of_all.items():
                if (other_lang, kg.lang) in seed_alignlinks:  # seed_alignlinks {(lang1, lang2): 2-col np.array}
                    # use if to avoid retrain the same pair of languages
                    align_links = seed_alignlinks[(other_lang, kg.lang)]
                    align_model.fit([align_links[:, 0], align_links[:, 1]], np.zeros(align_links.shape[0]),
                                    epochs=param.epoch2, batch_size=param.batch_size, shuffle=True, )

        # self-learning
        for kg in all_kgs:
            for other_kg in all_kgs:
                if other_kg.lang != kg.lang and (other_kg.lang, kg.lang) in seed_alignlinks:
                    print(f'self learning[{kg.lang}][{other_kg.lang}]')
                    seeds = seed_alignlinks[(other_kg.lang, kg.lang)]
                    found = extend_seed_align_links(other_kg, kg, seeds)
                    if len(found) > 0:  # not []
                        new_seeds = np.concatenate([seeds, found], axis=0)
                        seed_alignlinks[(other_kg.lang, kg.lang)] = new_seeds

        # train knowledge model
        kg0.model.fit([kg0.h_train, kg0.r_train, kg0.t_train], kg0.y_train,
                      epochs=param.epoch10, batch_size=param.batch_size, shuffle=True, )
        for kg1 in supporter_kgs:
            kg1.model.fit([kg1.h_train, kg1.r_train, kg1.t_train], kg1.y_train,
                          epochs=param.epoch11, batch_size=param.batch_size, shuffle=True, )

        model_weights = []
        if i % param.val_freq == 0:  # validation
            logging.info(f'=== round {i}')
            logging.info(f'[{kg0.lang}]')
            hits10_kg0 = validator.test(TestMode.KG0)


    kg0.save_model(model_dir)
    for kg1 in supporter_kgs:
        kg1.save_model(model_dir)

    save_model_structure(kg0.kNN_finder, os.path.join(model_dir, 'kNN_finder.json'))

    for kg1 in supporter_kgs:
        kg1.populate_alignlinks(kg0, seed_alignlinks)

    # Knowledge/alignment training is finished.  The original implementation
    # documents this flag as False in test mode so each supporter's reordered
    # embedding table is built once and then reused.  Leaving it at its module
    # default (True) rebuilds the same table for every individual prediction.
    param.updating_embedding = False

    choices = ['-val.tsv', '-test.tsv']  # '-val.tsv' if predict on validation data, '-test.tsv' if on test data
    validator = MultiModelTester(kg0, supporter_kgs)

    kg0.filtered_reordered_embedding_matrix = None
    for kg1 in supporter_kgs:
        kg1.filtered_reordered_embedding_matrix = None

    val_df = None
    test_df = None

    for suffix in choices:
        testfile = join(data_dir, target_lang + suffix)
        output = join(model_dir, 'results'+suffix)
        testcases = pd.read_csv(testfile, sep='\t', header=None).values[:args.MAX_SAM]
        param.n_test = testcases.shape[0]  # test on all training triples
        print('Loaded test cases from: %s'%testfile)

        results_df = validator.test_and_record_results(testcases)
        if suffix == '-val.tsv':
            val_df = results_df
        elif suffix == '-test.tsv':
            test_df = results_df

        results_df.to_csv(join(output), sep='\t', index=False)
