from csv2json_fg import csv2json_fg
from csv2json_bg import csv2json_bg

if __name__ == '__main__':
    label_csv = 'stage_2_train_labels.csv'
    csv2json_fg(label_csv)
    csv2json_bg(label_csv)

