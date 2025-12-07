import json
import spacy
import os
import argparse


def extract_keywords(_args):
    model = spacy.load('en_core_web_sm')

    with open(_args.file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for item in data:
            context = ""
            if _args.question:
                context = context + data[item]['question'] + ' '
            if _args.answer:
                context = context + data[item]['choices'][data[item]['answer']] + '. '
            if _args.caption:
                context = context + data[item]['caption']
            if _args.solution:
                context = context + data[item]['solution']

            doc = model(context)
            keywords = ', '.join([token.text for token in doc if not token.is_stop and not token.is_punct])
            data[item]["keywords"] = keywords
            if _args.show_result:
                print(f"Id: {item}\nContext: {context}\nKeywords:{keywords}\n")

    # with open(_args.output_file_path, 'w', encoding='utf-8') as OutFile:
    #     json.dump(data, OutFile, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--file_path', type=str, default='/cap_trainset.json')
    parse.add_argument('--output_file_path', type=str, default='/cap_trainset.json')
    parse.add_argument('--question', action='store_true', default=False)
    parse.add_argument('--answer', action='store_true', default=False)
    parse.add_argument('--caption', action='store_true', default=False)
    parse.add_argument('--solution', action='store_true', default=False)
    parse.add_argument('--show_result', action='store_true', default=False)
    args = parse.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    extract_keywords(args)

