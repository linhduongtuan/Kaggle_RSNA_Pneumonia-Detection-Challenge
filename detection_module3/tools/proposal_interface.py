import pickle
import numpy as np


def create_proposal(proposal_file):
    with open(proposal_file, 'rb') as f:
        proposals = pickle.load(f, encoding='bytes')
    boxes = np.array(proposals[b'boxes'])
    scores = np.array(proposals[b'scores'])
    names = np.array(proposals[b'names'])
    print(proposals.keys())

    return boxes, scores, names


if __name__ == '__main__':
    proposal_file = "/home/wccui/RSNA/data/cache/test_rpn_proposals.pkl"
    boxes, scores, names = create_proposal(proposal_file)
    print(boxes.shape)
    print(scores.shape)
    print(names.shape)
    print(boxes[0][0])
    print(scores[0][0])
    print(names[0])
