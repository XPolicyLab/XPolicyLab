# Multi-modal Database (MMDB)

The Multi-modal Database (MMDB) is a tool designed to help create and query a database of images used in a dataset mixture. Once the database is built, the tool also provides a web-based UI to search for similar images using a query image. Image similarity is measured using cosine similarity, based on features extracted by a pre-trained image encoder, such as SigLIP.

## Building the Database

To build the database, run the following command:

```bash
torchrun --nproc-per-node <nproc-per-node> tools/mmdb/build.py --mmdb-dir <mmdb-dir> --dataset <dataset> --model-name-or-path <model-name-or-path>
```

### Example:

```bash
torchrun --nproc-per-node 8 tools/mmdb/build.py --mmdb-dir mmdb --dataset vila-v1.5-sft --model-name-or-path google/siglip-so400m-patch14-384
```

This example builds the MMDB using 8 GPUs to process the dataset mixture `vila-v1.5-sft` with the model `siglip-so400m-patch14-384`, and stores the resulting database in the `mmdb` directory.

## Querying the Database

To query the database via the web UI, use the following command:

```bash
python tools/mmdb/app.py --mmdb-dir <mmdb-dir> --port <port>
```

This starts a web server on the specified `<port>`, allowing you to upload an image and search for similar images from the database.
