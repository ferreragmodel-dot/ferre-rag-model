from retrieval_utils import parse_filters

def main():
    f = ["type=lesson", 'doc="Notes_White shirt"', "year=1985"]
    where = parse_filters(f)
    assert where == {"type": "lesson", "doc": "Notes_White shirt", "year": 1985}
    print("PASS: parse_filters works:", where)

if __name__ == "__main__":
    main()