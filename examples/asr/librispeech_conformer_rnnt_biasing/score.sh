dir=$1 # the path to the decoding dir, e.g. experiments/librispeech_clean100_suffix600_tcpgen500_sche30_nodrop/decode_test_clean_b10_KB1000/
/home/gs534/rds/hpc-work/work/espnet-mm/tools/sctk-2.4.10/bin/sclite -r ${dir}/ref.trn.txt trn -h ${dir}/hyp.trn.txt trn -i rm -o all stdout > ${dir}/result.wrd.txt
