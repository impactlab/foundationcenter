import csv
import json

def main():
  output = {}
  with open('PCS_Subject_Taxonomy.csv', 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in csvreader:
      found_match = False
      for idx, _ in enumerate(row):
        # print idx
        # print row[idx]
        # print row[idx+1]
        if row[idx] and row[idx+1]:
          output[row[idx]] = row[idx+1].strip()
          found_match = True
          break

      if not found_match:
        assert False, "error parsing row: {}".format(row)

  with open('PCS_Subject_Taxonomy.json', 'w') as outfile:
    json.dump(output, outfile, sort_keys=True,indent=4, separators=(',', ': '))

if __name__ == "__main__":
  main()
