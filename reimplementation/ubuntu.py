"""
https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus

https://github.com/rkadlec/ubuntu-ranking-dataset-creator

https://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz

questions: what is the average number of turns? can we plot this? maybe there's a privacy there comparing many turns to one turn. :)

ah: "The conversations have an average of 8 turns each, with a minimum of 3 turns."

two person conversations :)

what does signal do with group conversations? i guess i want to keep that out of scope for now.

by the way, that first text with no to: field (since any employee can take up the chat): i'll ignore for now.

https://medium.com/analytics-vidhya/optimized-ways-to-read-large-csvs-in-python-ab2b36a7914e

"""

import ssd
import csv
from dateutil import parser
import logging
from datetime import datetime

# logging.basicConfig(filename='ubuntu.log', filemode='w', level=logging.DEBUG)

max_lines = 9300000
# max_lines = 100000

min_print = 8000
max_print = 8005


def parse_data():
    print(f"{datetime.now()} parse_data start")
    logging.info(f"{datetime.now()} parse_data")

    messages = []

    with open('../ubuntu/Ubuntu-dialogue-corpus/dialogueText_196.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        column_names = None
        date_index = None
        from_index = None
        to_index = None
        for (_, _, date, sender, receiver, message_text) in csv_reader:
            if line_count >= max_lines:
                break
            if line_count == 0:
                # column_names = row
                # print(column_names)
                # date_index = column_names.index("date")
                # from_index = column_names.index("from")
                # to_index = column_names.index("to")
                # text_index = column_names.index("text")
                line_count += 1
            else:
                message = ssd.Message(date, sender, receiver, text=message_text)
                messages.append(message)
                # messages.append((parser.parse(row[date_index]), row[from_index], row[to_index], row[text_index]))
                # access = (date, sender, receiver, message_text)
                line_count += 1
        logging.info(f'Processed {line_count} lines.')
        print(f'{datetime.now()} processed {line_count} lines')

    for i in range(min_print, max_print):
        logging.info(messages[i])

    return messages

# if __name__ == "__main__":
#     main()
