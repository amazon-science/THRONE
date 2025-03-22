# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

INSTRUCTION1 = "Read the text and answer the question."
INSTRUCTION2 = "Read the text about an image and answer the question."
INSTRUCTION3 = "Now answer this question."

Q_YN_S1a = "Please answer yes or no. Is there {0} {1} in this image?"
Q_YN_P1a = "Please answer yes or no. Are there {0} in this image?"
Q_YN_S1b = "Is there {0} {1} in this image?"
Q_YN_P1b = "Are there {0} in this image?"

Q_YN_S2a = "Please answer yes or no. Does the text imply {0} {1} is in the image?"
Q_YN_P2a = "Please answer yes or no. Does the text imply {0} are in the image?"
Q_YN_S2b = "Does the text imply {0} {1} is in the image?"
Q_YN_P2b = "Does the text imply {0} are in the image?"

Q_YN_S3a = "Please answer yes or no. Does the text give evidence for {0} {1} being present?"
Q_YN_P3a = "Please answer yes or no. Does the text give evidence for {0} being present?"
Q_YN_S3b = "Does the text give evidence for {0} {1} being present?"
Q_YN_P3b = "Does the text give evidence for {0} being present?"

Q_YN_S4a = "Please answer yes or no. Does the text explicitly mention {0} {1} is in the image?"
Q_YN_P4a = "Please answer yes or no. Does the text explicitly mention {0} are in the image?"
Q_YN_S4b = "Does the text explicitly mention {0} {1} is in the image?"
Q_YN_P4b = "Does the text explicitly mention {0} are in the image?"

Q_LIST = [
    [Q_YN_S1a, Q_YN_P1a],
    [Q_YN_S1b, Q_YN_P1b],
    [Q_YN_S2a, Q_YN_P2a],
    [Q_YN_S2b, Q_YN_P2b],
    [Q_YN_S3a, Q_YN_P3a],
    [Q_YN_S3b, Q_YN_P3b],
    [Q_YN_S4a, Q_YN_P4a],
    [Q_YN_S4b, Q_YN_P4b],
]

I_LIST = [
    INSTRUCTION1,
    INSTRUCTION2,
    INSTRUCTION3,
]
