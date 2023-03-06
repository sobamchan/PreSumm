import os
from argparse import ArgumentParser

import sienna
import spacy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["scitldr", "scisummnet"], type=str, required=True)
    parser.add_argument("--ddir", type=str, required=True)
    parser.add_argument("--odir", type=str, required=True)
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")

    splits = ["train", "val", "test"]

    if args.dataset == "scitldr":
        for split in splits:
            dpath = os.path.join(args.ddir, f"{split}.jsonl")
            data = sienna.load(dpath)

            new_data = []

            for x in data:
                new_x = {"src": [], "tgt": []}

                for sent in x["source"]:
                    new_x["src"].append([tok.text for tok in nlp(sent)])

                for sent in nlp(x["target"][0]).sents:
                    new_x["tgt"].append([tok.text for tok in sent])

                new_data.append(new_x)

            assert len(data) == len(new_data)

            opath_json = os.path.join(args.odir, f"{args.dataset}.{split}.0.json")
            sienna.save(new_data, opath_json)
            print(f"Saving to {opath_json}...")

    else:
        assert args.dataset == "scisummnet", "Not supported dataset."

        for split in splits:
            dpath = os.path.join(args.ddir, f"{split}.jsonl")
            data = sienna.load(dpath)

            new_data = []

            for x in data:
                new_x = {"src": [], "tgt": []}

                src = f'{x["abstract"]} {x["introduction"]} {x["conclusion"]}'
                for sent in nlp(src).sents:
                    new_x["src"].append([tok.text for tok in sent])

                for sent in x["summary_sents"]:
                    new_x["tgt"].append([tok.text for tok in nlp(sent)])

                new_data.append(new_x)

            assert len(data) == len(new_data)

            opath_json = os.path.join(args.odir, f"{args.dataset}.{split}.0.json")
            sienna.save(new_data, opath_json)
            print(f"Saving to {opath_json}...")
