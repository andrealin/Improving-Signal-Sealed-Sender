import ssd

import os
import logging

data_dir = '../maildir'


def parse_data():
    # from whom, to whom, timestamp
    senders = next(os.walk(data_dir))[1]
    ids = set()
    within = []
    users = set()
    for sender in senders:
        slice_index = sender.index("-")
        user = sender[:slice_index]
        users.add(user)
    print(users)
    for sender in senders:
        paths = []

        sent_dir = data_dir + "/" + sender + "/sent_items"

        try:
            filenames = next(os.walk(sent_dir))[2]
        except:
            # print("no sent items folder")
            continue
        for filename in filenames:
            path = sent_dir + "/" + filename
            paths.append(path)

        sent_dir = data_dir + "/" + sender + "/sent"

        try:
            filenames = next(os.walk(sent_dir))[2]
        except:
            # print("no sent items folder")
            continue
        for filename in filenames:
            path = sent_dir + "/" + filename
            paths.append(path)

        for path in paths:
            with open(path) as f:
                try:
                    id_value, timestamp, sender, receiver = [next(f) for i in range(4)]
                except Exception:
                    pass
                    # print("error reading")
                    # traceback.print_exc()
                if "," not in receiver:  # no group emails
                    if id_value not in ids:
                        timestamp = timestamp[6:]
                        ids.add(id)

                        try:
                            sender_slice_end = sender.index("@")
                            sender = sender[:sender_slice_end]
                        except:
                            # print(f"no at in sender {sender}")
                            continue

                        try:
                            sender_slice_start = sender.rindex(".")
                        except:
                            sender_slice_start = len("From: ")
                            # print("no period in sender address")
                        sender = sender[sender_slice_start + 1:]

                        try:
                            receiver_slice_end = receiver.index("@")
                            receiver = receiver[:receiver_slice_end]
                        except:
                            pass

                        try:
                            receiver_slice_start = receiver.rindex(".")
                        except:
                            receiver_slice_start = len("To: ")
                            # print("no period in receiver address")
                        receiver = receiver[receiver_slice_start + 1:]

                        if sender in users and receiver in users:
                            # print(timestamp, sender, receiver)
                            within.append(ssd.Message(timestamp, sender, receiver))
                    else:
                        pass
                        # print("id already added")

    logging.info(f"total number of emails {len(ids)})")
    logging.info(f"number of emails used {len(within)})")

    return users, within
