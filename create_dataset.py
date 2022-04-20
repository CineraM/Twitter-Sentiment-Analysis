def writeAll(folder):
    f_test = open("datasetCreation/emotions_{}.csv".format(folder),
                  "w", encoding="utf-8")
    count = 0

    def writeFile(filename, folder, f_test, count):
        with open("datasetCreation/{}/{}".format(folder, filename), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    f_test.write("{},{},{}".format(
                        count, filename.split(".")[0], line.encode('cp1252').decode('utf-8')))
                    count += 1
                except:
                    pass
        return count

    count = writeFile("joy.csv", folder, f_test, count)
    count = writeFile("sadness.csv", folder, f_test, count)
    count = writeFile("fear.csv", folder, f_test, count)
    count = writeFile("anger.csv", folder, f_test, count)

    f_test.close()


writeAll("test")
writeAll("train")
