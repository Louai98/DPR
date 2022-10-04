> ### The names of the dataset files are squad and dpr_wiki but the information into the files is from IDDRS. I did that to match the values in the configuration file

## Retriever training
```bash
python train_dense_encoder.py train_datasets=[squad1_train] dev_datasets=[squad1_dev] train=biencoder_local output_dir=output
```

## Retriever inference

Generating representation vectors for the static documents dataset is a highly parallelizable process which can take up to a few days if computed on a single GPU. You might want to use multiple available GPU servers by running the script on each of them independently and specifying their own shards.

```bash
python generate_dense_embeddings.py \
	model_file={path to biencoder checkpoint} \
	ctx_src= dpr_wiki \
	shard_id={shard_num, 0-based} num_shards={total number of shards} \
	out_file={result files location + name PREFX}	
```

