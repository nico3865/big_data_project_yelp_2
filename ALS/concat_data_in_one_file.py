





first_file0 = open("review_50K_0.json")
first_file1 = open("review_50K_1.json")
first_file2 = open("review_50K_2.json")
first_file3 = open("review_50K_3.json")
first_file4 = open("review_50K_4.json")
first_file5 = open("review_50K_5.json")
first_file6 = open("review_50K_6.json")
first_file7 = open("review_50K_7.json")
first_file8 = open("review_50K_8.json")
first_file9 = open("review_50K_9.json")
first_file10 = open("review_50K_10.json")
first_file11 = open("review_50K_11.json")
first_file12 = open("review_50K_12.json")
first_file13 = open("review_50K_13.json")
first_file14 = open("review_50K_14.json")
first_file15 = open("review_50K_15.json")
first_file16 = open("review_50K_16.json")
first_file17 = open("review_50K_17.json")
first_file18 = open("review_50K_18.json")



# read lines
# first_lines = first_file.readlines()
# second_lines = second_file.readlines()

first_file_lines0 = first_file0.readlines()
first_file_lines1 = first_file1.readlines()
first_file_lines2 = first_file2.readlines()
first_file_lines3 = first_file3.readlines()
first_file_lines4 = first_file4.readlines()
first_file_lines5 = first_file5.readlines()
first_file_lines6 = first_file6.readlines()
first_file_lines7 = first_file7.readlines()
first_file_lines8 = first_file8.readlines()
first_file_lines9 = first_file9.readlines()
first_file_lines10 = first_file10.readlines()
first_file_lines11 = first_file11.readlines()
first_file_lines12 = first_file12.readlines()
first_file_lines13 = first_file13.readlines()
first_file_lines14 = first_file14.readlines()
first_file_lines15 = first_file15.readlines()
first_file_lines16 = first_file16.readlines()
first_file_lines17 = first_file17.readlines()
first_file_lines18 = first_file18.readlines()

all_lines = first_file_lines0 + first_file_lines1 + first_file_lines2 + first_file_lines3 + first_file_lines4 + first_file_lines5 + first_file_lines6 + first_file_lines7 + first_file_lines8 + first_file_lines9 + first_file_lines10 + first_file_lines11 + first_file_lines12 + first_file_lines13 + first_file_lines14 + first_file_lines15 + first_file_lines16 + first_file_lines17 + first_file_lines18


textdoc = open("review_CONCAT.json", "w")
textdoc.writelines(all_lines)
