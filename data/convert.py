import csv
import re

input_file = r"F:\AOOB-Graph\data\Full_alarms_with_result.csv"
output_file = r"F:\AOOB-Graph\data\Full_alarms.csv"

with open(input_file, "r", encoding="utf-8-sig", newline="") as fin, \
     open(output_file, "w", encoding="utf-8", newline="") as fout:

    # Preserve/keep the "sep=;" line as-is
    first_line = fin.readline()
    fout.write(first_line.strip() + "\n")

    reader = csv.DictReader(fin, delimiter=";")
    fieldnames = reader.fieldnames

    writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()

    for row in reader:
        # Clear Classification and Comment
        row["Classification"] = ""
        row["Comment"] = ""

        # Strip the leading "ALARM (A) array_out_of_bounds: " (or similar) from Message
        msg = row["Message"]
        if ":" in msg:
            # Keep only the part after the last ": " that precedes the actual detail
            msg = msg.split(": ", 1)[1] if ": " in msg else msg
        row["Message"] = msg.strip()

        writer.writerow(row)

print(f"Done. Written to: {output_file}")