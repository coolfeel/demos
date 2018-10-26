
# import re
#
#
# def textParse(bigString):
#     listOfTokens = re.split(r'\W*', bigString)
#     return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#
#
# def spamTest():
#     docList = []
#     classList = []
#     fullText = []
#     for i in range(1, 26):
#         wordList = textParse(open('email/spam/%d.txt' % i).read())