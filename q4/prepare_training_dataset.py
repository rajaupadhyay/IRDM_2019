'''
Build the training dataset for logit

First we're just testing the model on 6000 claims

- Iterate over the claimToDocsDict
- For each claim and respective docs list
    - First get the relevant sentences from the evidence field for the specific claim (Positive examples)
    - Randomly sample the same number (+1/2) irrelevant sentences from the docs list
    - Represent these sentences using embeddings (Glove Gensim)
    - Represent the claim using the same embedding method
    - A single datapoint would now consist of one sentence (relevant or irrelevant) and the claim (as 2 distinct features represented as embeddings (,300))
    - The label for the datapoint would be whether the sentence (1st feature) is relevant or not with respect to the claim (2nd feature)


TBD
- Put claims in DB
'''
