import gzip

from preprocess import *
from util.file_util import write_lines
from util.list_util import rindex


def preprocess_kp20k_test():
    kp20k = load_dataset(os.path.join(".", "kp20k", "kp20k_test.json"))
    with open(os.path.join("kp20k", "kp20k_test_kws.json"), "w+", encoding="UTF-8") as fp_kw, \
            open(os.path.join("kp20k", "kp20k_test_no_kws.json"), "w+", encoding="UTF-8") as fp_nkw:
        for idx, r in enumerate(kp20k):
            kp20k[idx]["keywords"] = r["keywords"].split(";")
            kp20k_json = json.dumps(kp20k[idx])
            fp_kw.write(str(kp20k_json))
            fp_kw.write("\n")
            kp20k[idx]["keywords"] = []
            kp20k_json = json.dumps(kp20k[idx])
            fp_nkw.write(str(kp20k_json))
            fp_nkw.write("\n")


def preprocess_kp20k_train_debug():
    kp20k = load_dataset(os.path.join(".", "kp20k", "kp20k_train.json"))
    with open(os.path.join("kp20k", "kp20k_train_debug_kws.json"), "w+", encoding="UTF-8") as fp_kw, \
            open(os.path.join("kp20k", "kp20k_train_debug_no_kws.json"), "w+", encoding="UTF-8") as fp_nkw:
        for idx, r in enumerate(kp20k):
            kp20k_json = json.dumps(kp20k[idx])
            fp_kw.write(str(kp20k_json))
            fp_kw.write("\n")
            kp20k[idx]["keywords"] = []
            kp20k_json = json.dumps(kp20k[idx])
            fp_nkw.write(str(kp20k_json))
            fp_nkw.write("\n")
            if idx >= 1000:
                return


def preprocess_semeval_test():
    lines = read_lines(os.path.join(".", "semeval", "semeval_test.json"))
    lines = [line.replace("\\n", " ") for line in lines]
    write_lines(os.path.join(".", "semeval", "semeval_test_rm_break.json"), lines, "w+")
    semeval = load_dataset(os.path.join(".", "semeval", "semeval_test_rm_break.json"))
    with open(os.path.join("semeval", "semeval_test_kws.json"), "w+", encoding="UTF-8") as fp_kw, \
            open(os.path.join("semeval", "semeval_test_no_kws.json"), "w+", encoding="UTF-8") as fp_nkw:
        for idx, r in enumerate(semeval):
            semeval[idx]["keywords"] = r["keywords"].split(";")
            kp20k_json = json.dumps(semeval[idx])
            fp_kw.write(str(kp20k_json))
            fp_kw.write("\n")
            semeval[idx]["keywords"] = []
            kp20k_json = json.dumps(semeval[idx])
            fp_nkw.write(str(kp20k_json))
            fp_nkw.write("\n")
    pass


def preprocess_meng(input_file, output_file):
    dataset_in = [line["meng17_tokenized"] for line in load_dataset(input_file)]
    with open(output_file, "w+", encoding="UTF-8") as fp:
        for d in tqdm(dataset_in):
            j = json.dumps({
                "title": "",
                "abstract": " ".join(d["src"]),
                "keywords": [" ".join(k) for k in d["tgt"]]
            })
            fp.write(str(j))
            fp.write("\n")
    pass


def preprocess_test(input_file, output_file_kws, output_file_nkws):
    kp20k = load_dataset(input_file)
    with open(output_file_kws, "w+", encoding="UTF-8") as fp_kw, \
            open(output_file_nkws, "w+", encoding="UTF-8") as fp_nkw:
        for idx, r in enumerate(kp20k):
            kp20k_json = json.dumps(kp20k[idx])
            fp_kw.write(str(kp20k_json))
            fp_kw.write("\n")
            kp20k[idx]["keywords"] = []
            kp20k_json = json.dumps(kp20k[idx])
            fp_nkw.write(str(kp20k_json))
            fp_nkw.write("\n")


def preprocess_liu(input_content_file, input_keyword_file, output_file_kws):
    content = read_lines(input_content_file)
    keyword = [i["keywords"] for i in load_dataset(input_keyword_file)]
    assert len(content) == len(keyword)
    with open(output_file_kws, "w+", encoding="UTF-8") as fp_kw:
        for idx, (c, k) in enumerate(zip(content, keyword)):
            kp20k_json = json.dumps({
                "title": "",
                "abstract": c.replace(" ##", ""),
                "keywords": k,
            })
            fp_kw.write(str(kp20k_json))
            fp_kw.write("\n")


def preprocess_liu_train(input_data_path, input_liu_file, output_path):
    in_lines = read_lines(os.path.join(input_data_path, "seq.in"))
    out_lines = read_lines(os.path.join(input_data_path, "seq.out"))
    present_lines = read_lines(os.path.join(input_data_path, "seq.labelp"))
    center_lines = read_lines(os.path.join(input_data_path, "seq.labelc"))
    liu_lines = read_lines(input_liu_file)
    for idx, (p, lp) in tqdm(enumerate(zip(present_lines, liu_lines))):
        p = p.split(" ")
        # lp = lp.replace("X", "I")
        lp = lp.split(" ")
        temp = p[:rindex(p, "N") + 1] + lp
        assert len(temp) == len(p)
        present_lines[idx] = " ".join(temp)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_lines(os.path.join(output_path, "seq.in"), in_lines, "w+")
    write_lines(os.path.join(output_path, "seq.out"), out_lines, "w+")
    write_lines(os.path.join(output_path, "seq.labelp"), present_lines, "w+")
    write_lines(os.path.join(output_path, "seq.labelc"), center_lines, "w+")


def sample_data(input_file, output_file, num):
    lines = read_lines(input_file)
    random.shuffle(lines)
    write_lines(output_file, lines[:num], "w+")


def preprocess_acm(input_file, output_file_kw, output_file_nkw):
    def extract_content(tagged_line, tag):
        """Extract content from SGML line."""
        return tagged_line.replace("<" + tag + ">", "").replace("</" + tag + ">", "")

    title = ""
    abstract = ""
    keywords = []

    with gzip.open(input_file, 'rt', encoding="utf-8") as fin, \
            open(output_file_kw, "w+", encoding="UTF-8") as fw_kw, \
            open(output_file_nkw, "w+", encoding="UTF-8") as fw_nkw:
        for i, line in enumerate(tqdm(fin)):
            line = line.strip()
            if line.startswith('<DOC>'):
                is_in_document = True
                pass

            elif line.startswith('<DOCNO>'):
                doc_id = extract_content(line, "DOCNO")
                pass

            elif line.startswith('<TITLE>'):
                title = extract_content(line, "TITLE")

            elif line.startswith('<TEXT>'):
                abstract = extract_content(line, "TEXT")

            elif line.startswith('<HEAD>'):
                keywords = extract_content(line, "HEAD")
                keywords = [k.strip() for k in keywords.split("//")]

            elif line.startswith('</DOC>'):
                if keywords:
                    data_line = json.dumps({
                        "title": title,
                        "abstract": abstract,
                        "keywords": keywords,
                    })
                    fw_kw.write(str(data_line))
                    fw_kw.write("\n")

                    data_line = json.dumps({
                        "title": title,
                        "abstract": abstract,
                        "keywords": [],
                    })
                    fw_nkw.write(str(data_line))
                    fw_nkw.write("\n")

                title = ""
                abstract = ""
                keywords = []



