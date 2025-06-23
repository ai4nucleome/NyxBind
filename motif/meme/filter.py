import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check Tomtom matches and write results to CSV.")
    parser.add_argument("--name", type=str, required=True, help="Project or method name (used in folder and output naming)")
    parser.add_argument("--id_file", type=str, required=True, help="Path to motif_filtered.txt")
    parser.add_argument("--tom_result", type=str, required=True, help="Path to tomtom.tsv")
    parser.add_argument("--output_dir", type=str, default="./res", help="Output directory")

    args = parser.parse_args()

    output_file = os.path.join(args.output_dir, args.name + ".csv")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tomtom results
    tom_rows = []
    with open(args.tom_result, "r") as tom_file:
        reader = csv.DictReader(tom_file, delimiter="\t")
        for row in reader:
            tom_rows.append(row)

    if tom_rows:
        base_fields = list(tom_rows[0].keys())
    else:
        base_fields = ['Query_ID', 'Target_ID']

    if 'status' not in base_fields:
        fieldnames = base_fields[:2] + ['status'] + base_fields[2:]
    else:
        fieldnames = base_fields

    unmatch_count = 0

    with open(output_file, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        with open(args.id_file, "r") as f:
            next(f)  # skip header
            for line in f:
                fields = line.strip().split(",")
                if len(fields) < 2:
                    continue
                tf_name = fields[0].strip()
                tf_id = fields[1].strip()
                matched_rows = [
                    row for row in tom_rows
                    if row['Query_ID'].strip() == tf_name and row['Target_ID'].strip() == tf_id
                ]

                if matched_rows:
                    for match in matched_rows:
                        try:
                            q_value = float(match.get('q-value', 1.0))
                        except ValueError:
                            q_value = 1.0

                        if q_value <= 0.01:
                            match['status'] = 'matched'
                        else:
                            match['status'] = 'unmatch'
                            unmatch_count += 1
                            print(f"[High Q-VALUE] {tf_name} ({tf_id}) -> q-value: {q_value}")

                        writer.writerow(match)
                else:
                    print(f"[NO MATCH] {tf_name} ({tf_id})")
                    unmatch_count += 1
                    row_data = {key: '' for key in fieldnames}
                    row_data['Query_ID'] = tf_name
                    row_data['Target_ID'] = tf_id
                    row_data['status'] = 'unmatch'
                    writer.writerow(row_data)

    print(f"\nTotal unmatched motifs: {unmatch_count}")


if __name__ == "__main__":
    main()
