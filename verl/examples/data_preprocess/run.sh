# python longcontext_qa_extract.py --input_path /user/zhangyixuan/data/raw_sft_data/qa_en/ --local_dir /user/xuxiaoyue/rldata/qa_en/ --test_size 0.15 --max_limit 12000
# python longcontext_qa_direct.py --input_path /user/xuxiaoyue/synthetic_longsft_data/sampled-0714/12k-20k/qasper_magiccc_2000/jsonfile_magiccc_2000_4/ --local_dir /user/xuxiaoyue/rldata/qa_magic/ --test_size 0.15
python longcontext_qa_direct.py --input_path /user/xuxiaoyue/synthetic_longsft_data/sampled-0714/12k-20k/qasper_tailor_500/jsonfile_tailor_500_4/ --local_dir /user/xuxiaoyue/rldata/qa_tailo2/ --test_size 0.15
# python longcontext_qa_extract.py --input_path /user/zhangyixuan/data/raw_sft_data/retrieval_en/ --local_dir /user/xuxiaoyue/rldata/retrieval_en/ --test_size 0.15 --max_limit 12000
# python longcontext_qa_extract.py --input_path /user/zhangyixuan/data/raw_sft_data/math_find/ --local_dir /user/xuxiaoyue/rldata/math_find/ --test_size 0.15 --source math_find
# python longcontext_qa_direct.py --input_path /user/zhangyixuan/data/raw_sft_data/en_long_align_like_8_64_new/ --local_dir /user/xuxiaoyue/rldata/en_like/ --test_size 0.15 --max_limit 37000


# python longcontext_qa_extract_no_noinfo.py --input_path /user/zhangyixuan/data/raw_sft_data/qa_en/ --local_dir /user/xuxiaoyue/rldata/qa_en_0/ --test_size 0.15 --max_limit 12000
# python longcontext_qa_extract_no_noinfo.py --input_path /user/zhangyixuan/data/raw_sft_data/retrieval_en/ --local_dir /user/xuxiaoyue/rldata/retrieval_en_0/ --test_size 0.15 --max_limit 12000