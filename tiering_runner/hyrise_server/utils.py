def query_to_one_line(query):
    single_line = query.replace("\n", " ")
    return ' '.join(single_line.split())
