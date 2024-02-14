#!/usr/bin/env python3
import sys
import csv
import json

def usage(name, err):
    sys.stderr.write("\nUSAGE:\n  %s <csv-map-file> <json-curve-file> [<json-curve-file> ...]\n\n" % name)
    sys.exit(err)


def main():
    if len(sys.argv) < 3:
        usage(sys.argv[0], 1)

    curve_fnames = sys.argv[2:]

    csv_map = {}
    list_of_column_names = []
    with open(sys.argv[1], "r") as map_file:

        csv_reader = csv.reader(map_file, delimiter=',')

        # loop to iterate through the rows of csv
        for row in csv_reader:
            if list_of_column_names == []:
                if row[0].startswith('#'):
                    continue
                list_of_column_names = row
                lon_pos = next(i for i, x in enumerate(
                    list_of_column_names) if x == 'lon')
                lat_pos = next(i for i, x in enumerate(
                    list_of_column_names) if x == 'lat')
                continue

            csv_map[(float(row[lon_pos]), float(row[lat_pos]))] = row

        for curve_fname in curve_fnames:
            with open(curve_fname, "r") as curve_file:
                curve_json = json.load(curve_file)
                for item in curve_json['features']:
                    if (item['properties']['lon'],
                        item['properties']['lat']) not in csv_map.keys():
                        print('WARNING: item with coords (lon, lat) (%s, %s) not found.' % (
                            item['properties']['lon'],
                            item['properties']['lat'],))

main()
