from gensim import utils

num_topics = 26

def read_doctopics(fname, eps=1e-6, renorm=True):
    mallet_version = "2.0.8"
    with utils.open(fname, 'rb') as fin:
        for lineno, line in enumerate(fin):
            if lineno == 0 and line.startswith(b"#doc "):
                continue  # skip the header line if it exists

            parts = line.split()[2:]  # skip "doc" and "source" columns

            # the MALLET doctopic format changed in 2.0.8 to exclude the id,
            # this handles the file differently dependent on the pattern
            if len(parts) == 2 * num_topics:
                doc = [
                    (int(id_), float(weight)) for id_, weight in zip(*[iter(parts)] * 2)
                    if abs(float(weight)) > eps
                ]
            elif len(parts) == num_topics and mallet_version != '2.0.7':
                doc = [(id_, float(weight)) for id_, weight in enumerate(parts) if abs(float(weight)) > eps]
            else:
                raise RuntimeError("invalid doc topics format at line %i in %s" % (lineno + 1, fname))

            if renorm:
                # explicitly normalize weights to sum up to 1.0, just to be sure...
                total_weight = float(sum(weight for _, weight in doc))
                if total_weight:
                    doc = [(id_, float(weight) / total_weight) for id_, weight in doc]
            yield doc

def inference():
    result = list(read_doctopics('../RQ3_results/post2readme_topics.infer'))
    with open('../RQ3_results/post2readme_doc_topics.txt', 'w') as f:
        for i, res in enumerate(result):
            f.write(f'{i},{res}\n')

inference()