#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Sequence
from typing import Any
from typing import Iterable
import json

import pandas as pd

__here__ = Path(__file__).absolute().parent
sys.path.append(str(__here__.parent))

_ACTION_VERB_MULTIPLIER = 1000


LOG = logging.getLogger("evaluate_action_anticipation")

class ValidationException(Exception):
    pass

class MissingPropertyException(ValidationException):
    def __init__(self, property: str, uid: int = None) -> None:
        self.property = property
        self.uid = uid

    def __str__(self):
        message = "Missing '{}' property".format(self.property)
        if self.uid is not None:
            message += " for uids {}.".format(self.uid)
        return message


class UnsupportedSubmissionVersionException(ValidationException):
    def __init__(self, valid_versions: Iterable[str], ver: str) -> None:
        self.valid_versions = valid_versions
        self.version = ver

    def __str__(self):
        return "Submission version '{}' is not supported, valid versions: {}".format(
            self.version, ", ".join(self.valid_versions)
        )


class UnsupportedChallengeException(ValidationException):
    def __init__(self, valid_challenges: Iterable[str], challenge: str) -> None:
        self.valid_challenges = valid_challenges
        self.challenge = challenge

    def __str__(self):
        return "Challenge '{}' is not supported, valid challenges: {}".format(
            self.challenge, ", ".join(self.valid_challenges)
        )


class MissingResultException(ValidationException):
    def __init__(self, ids: np.array) -> None:
        self.ids = ids

    def __str__(self):
        return (
            "The following narration ids are missing: "
            + ", ".join(self.ids.astype(str))
            + "."
        )


class UnexpectedResultException(ValidationException):
    def __init__(self, ids: np.array) -> None:
        self.ids = ids

    def __str__(self):
        return (
            "Found the following unexpected narration ids: "
            + ", ".join(self.ids.astype(str))
            + "."
        )


class InvalidClassEntry(ValidationException):
    def __init__(self, task: str, uid: int, invalid_entry: str) -> None:
        self.task = task
        self.uid = uid
        self.invalid_entry = invalid_entry

    def __str__(self):
        return "Found invalid {} class '{}' in segment {}".format(
            self.task, str(self.invalid_entry), self.uid
        )


class MissingScoreException(ValidationException):
    def __init__(self, entry_type: str, uid: int, missing_classes: np.ndarray) -> None:
        self.entry_type = entry_type
        self.uid = uid
        self.missing_classes = missing_classes

    def __str__(self):
        return "The following {} scores are not included for uid {}: {}.".format(
            self.entry_type, self.uid, ", ".join(self.missing_classes.astype(str))
        )


class UnexpectedScoreEntriesException(ValidationException):
    def __init__(self, task: str, uid: int, unexpected_classes: np.ndarray) -> None:
        self.task = task
        self.uid = uid
        self.unexpected_classes = unexpected_classes

    def __str__(self):
        return "Found the following unexpected {} entries for uid {}: {}.".format(
            self.task, self.uid, ", ".join(self.unexpected_classes.astype(str))
        )


class InvalidNumberOfActionScoresException(ValidationException):
    def __init__(self, uid: str, expected_count: int, actual_count: int) -> None:
        self.uid = uid
        self.expected_count = expected_count
        self.actual_count = actual_count

    def __str__(self):
        return (
            "The number of action scores provided for segment {} should be equal to {} "
            "but found {} scores."
        ).format(self.uid, self.expected_count, self.actual_count)


class InvalidActionIdException(ValidationException):
    def __init__(self, action: str, uid: int) -> None:
        self.action = action
        self.uid = uid

    def __str__(self):
        return "Action key {} in entry {} is not a correct 'verb,noun' id.".format(
            self.action, self.uid
        )


class InvalidScoreException(ValidationException):
    def __init__(self, task: str, uid: int, cls: str, score) -> None:
        self.task = task
        self.uid = uid
        self.cls = cls
        self.score = score

    def __str__(self):
        return (
            "Could not deserialize {} class '{}' score to float from segment {},"
            " it's value was '{}'"
        ).format(self.task, self.cls, self.uid, self.score)


class InvalidSLSException(ValidationException):
    def __init__(self, pt: int, tl: int, td: int):
        """
        Args:
            pt: Pretraining level
            tl: Training Labels level
            td: Training Data level
        """
        self.pt = pt
        self.tl = tl
        self.td = td

    def __str__(self):
        return (
            f"Invalid SLS: (PT = {self.pt}, TD = {self.td}, TL = {self.tl}). All "
            f"levels must be between 0 and 5."
        )


def validate_submission_challenge(submission, supported_challenges):
    if "challenge" not in submission.keys():
        raise MissingPropertyException("challenge")
    if submission["challenge"] not in supported_challenges:
        raise UnsupportedChallengeException(
            supported_challenges, submission["challenge"]
        )


def validate_submission_version(submission, valid_versions):
    if "version" not in submission.keys():
        raise MissingPropertyException("version")
    if submission["version"] not in valid_versions:
        raise UnsupportedSubmissionVersionException(
            valid_versions, submission["version"]
        )


def validate_supervision_level(submission):
    sls_properties = ["sls_pt", "sls_tl", "sls_td"]
    for property in sls_properties:
        if property not in submission:
            raise MissingPropertyException(property)
    for property in sls_properties:
        if not (0 <= submission[property] <= 5):
            raise InvalidSLSException(
                pt=submission["sls_pt"],
                tl=submission["sls_tl"],
                td=submission["sls_td"],
            )


def validate_submission(
    submission: Dict,
    expected_narration_ids: List[str],
    num_verbs: int,
    num_nouns: int,
    num_actions: int,
    entries_to_validate=-1,
    valid_versions: Tuple[str, ...] = ("0.2",),
    supported_challenges: Tuple[str, ...] = (
        "action_recognition",
        "action_anticipation",
    ),
):
    """Validates a submission
    Parameters:
    -----------
    submission
        deserialized json containing the submission
    expected_narration_ids
        the list of narration_ids which should be present in the submission
    num_verbs
        number of verbs predictions per test segment which should be included in the submission
    num_nouns
        number of nouns predictions per test segment which should be included in the submission
    num_actions
        number of action predictions per test segment which should be included in the submission
    entries_to_validate
        number of entries to validate before considering the submission valid, -1 indicates all entries.
    valid_versions
        list of valid versions
    supported_challenges
        list of challenges supported by scoring program

    """

    validate_submission_version(submission, valid_versions)
    validate_submission_challenge(submission, supported_challenges)
    validate_supervision_level(submission)

    if "results" not in submission.keys():
        raise MissingPropertyException("results")

    submission_ids = np.array(list(submission["results"].keys()))
    missing_ids = np.setdiff1d(expected_narration_ids, submission_ids)
    unexpected_ids = np.setdiff1d(submission_ids, expected_narration_ids)

    if len(missing_ids) != 0:
        raise MissingResultException(missing_ids)

    if len(unexpected_ids) > 0:
        raise UnexpectedResultException(unexpected_ids)

    task_classes = {
        "verb": np.arange(num_verbs),
        "noun": np.arange(num_nouns),
    }

    def validate_task_entry(scores_dict, uid, task):
        if task in scores_dict:
            class_entries = list(scores_dict[task].keys())
            try:
                seen_classes = np.array(class_entries, dtype=int)
            except Exception as e:
                for class_entry in class_entries:
                    try:
                        float(class_entry)
                    except ValueError:
                        raise InvalidClassEntry(task, uid, class_entry)
                raise e

            missing_classes = np.setdiff1d(task_classes[task], seen_classes)
            unexpected_classes = np.setdiff1d(seen_classes, task_classes[task])

            if len(missing_classes) > 0:
                raise MissingScoreException(task, uid, missing_classes)

            if len(unexpected_classes) > 0:
                raise UnexpectedScoreEntriesException(task, uid, unexpected_classes)

            for cls, score in scores_dict[task].items():
                if not isinstance(score, (int, float)):
                    raise InvalidScoreException(task, uid, cls, score)
        else:
            raise MissingPropertyException(task, uid)

    def validate_action_entry(uid, scores_dict):
        actions = np.array(list(scores_dict["action"].keys()))
        if len(actions) != num_actions:
            raise InvalidNumberOfActionScoresException(uid, num_actions, len(actions))

        for action, score in scores_dict["action"].items():
            try:
                verb, noun = [int(x) for x in action.split(",")]
            except Exception:
                raise InvalidClassEntry("action", uid, action)

            if verb not in task_classes["verb"] or noun not in task_classes["noun"]:
                raise InvalidActionIdException(action, uid)

            if not isinstance(score, (int, float)):
                raise InvalidScoreException("action", uid, action, score)

    has_actions = None
    for i, (uid, scores_dict) in enumerate(submission["results"].items()):
        if has_actions is None:
            has_actions = "action" in scores_dict
        if entries_to_validate == -1 or i < entries_to_validate:
            validate_task_entry(scores_dict, uid, "verb")
            validate_task_entry(scores_dict, uid, "noun")
            if has_actions:
                validate_action_entry(uid, scores_dict)
            else:
                if "action" in scores_dict:
                    raise ValueError(
                        "You must provide action submissions for all segments or no segments, "
                        "you can't only provide some."
                    )
        else:
            break

def action_id_from_verb_noun(verb: Union[int, np.array], noun: Union[int, np.array]):
    """
    Examples:
    >>> action_id_from_verb_noun(0, 0)
    0
    >>> action_id_from_verb_noun(0, 1)
    1
    >>> action_id_from_verb_noun(0, 351)
    351
    >>> action_id_from_verb_noun(1, 0)
    1000
    >>> action_id_from_verb_noun(1, 1)
    1001
    >>> action_id_from_verb_noun(np.array([0, 1, 2]), np.array([0, 1, 2]))
    array([   0, 1001, 2002])
    """
    return verb * _ACTION_VERB_MULTIPLIER + noun


def print_metrics(metrics):
    for name, value in metrics.items():
        print("{name}: {value:0.2f}".format(name=name, value=value))


def read_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def read_array_from_dict(array_dict: Dict):
    """
    Examples:
        >>> read_array_from_dict({'0': 1, '1': 2, '2': 3})
        array([1., 2., 3.])
    """
    keys = array_dict.keys()
    array = np.zeros(len(keys))
    for key in keys:
        array[int(key)] = array_dict[key]
    return array


def softmax(xs):
    """
    Examples:
        >>> res = softmax(np.array([0, 200, 10]))
        >>> np.sum(res)
        1.0
        >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
        True
        >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
        >>> np.sum(res, axis=1)
        array([1., 1., 1.])
        >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
        >>> np.sum(res, axis=1)
        array([1., 1.])
    """
    if xs.ndim == 1:
        xs = xs.reshape((1, -1))
    max_x = np.max(xs, axis=1).reshape((-1, 1))
    exp_x = np.exp(xs - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

def top_scores(scores: np.ndarray, top_n: int = 100):
    """
    Examples:
        >>> top_scores(np.array([0.2, 0.6, 0.1, 0.04, 0.06]), top_n=3)
        (array([1, 0, 2]), array([0.6, 0.2, 0.1]))
    """
    if scores.ndim == 1:
        top_n_idx = scores.argsort()[::-1][:top_n]
        return top_n_idx, scores[top_n_idx]
    else:
        top_n_scores_idx = np.argsort(scores)[:, ::-1][:, :top_n]
        top_n_scores = scores[
            np.arange(0, len(scores)).reshape(-1, 1), top_n_scores_idx
        ]
        return top_n_scores_idx, top_n_scores

def compute_action_scores(verb_scores, noun_scores, top_n=100):
    top_verbs, top_verb_scores = top_scores(verb_scores, top_n=top_n)
    top_nouns, top_noun_scores = top_scores(noun_scores, top_n=top_n)
    top_verb_probs = softmax(top_verb_scores)
    top_noun_probs = softmax(top_noun_scores)
    action_probs_matrix = (
        top_verb_probs[:, :, np.newaxis] * top_noun_probs[:, np.newaxis, :]
    )
    instance_count = action_probs_matrix.shape[0]
    action_ranks = action_probs_matrix.reshape(instance_count, -1).argsort(axis=-1)[
        :, ::-1
    ]
    verb_ranks_idx, noun_ranks_idx = np.unravel_index(
        action_ranks[:, :top_n], shape=(action_probs_matrix.shape[1:])
    )

    segments = np.arange(0, instance_count).reshape(-1, 1)
    return (
        (top_verbs[segments, verb_ranks_idx], top_nouns[segments, noun_ranks_idx]),
        action_probs_matrix.reshape(instance_count, -1)[
            segments, action_ranks[:, :top_n]
        ],
    )

def read_action_entries(dict: Dict):
    """
    Examples:
        >>> read_action_entries({'2,37': 0.85, '5,74': 0.15})
        {(2, 37): 0.85, (5, 74): 0.15}
    """
    actions = {}
    for id, score in dict.items():
        verb, noun = id.split(",")
        verb, noun = int(verb), int(noun)
        actions[(verb, noun)] = score
    return actions


def convert_results(
    narration_ids: Sequence[str], results_dict: Dict[str, Dict]
) -> Dict[str, np.ndarray]:
    converted_results = {}
    first_result = results_dict[narration_ids[0]]
    has_actions = "action" in first_result
    if not has_actions:
        LOG.info("Actions aren't present, computing from verb and noun scores")
    for narration_id, result in results_dict.items():
        verb_array = read_array_from_dict(result["verb"])
        noun_array = read_array_from_dict(result["noun"])
        if has_actions:
            if "action" in result:
                action_entries = {
                    action_id_from_verb_noun(verb, noun): score
                    for (verb, noun), score in read_action_entries(
                        result["action"]
                    ).items()
                }
                entry = {
                    "verb": verb_array,
                    "noun": noun_array,
                    "action": action_entries,
                }
            else:
                raise ValueError("Submit action predictions for each segment")
        else:
            entry = {"verb": verb_array, "noun": noun_array}
        converted_results[narration_id] = entry

    def collate_dict(dict: Dict, keys: Any, entry: Any) -> List:
        return [dict[key][entry] for key in keys]

    collated_results = {
        "verb": np.array(collate_dict(converted_results, narration_ids, "verb")),
        "noun": np.array(collate_dict(converted_results, narration_ids, "noun")),
    }

    if has_actions:
        collated_results["action"] = collate_dict(
            converted_results, narration_ids, "action"
        )
    else:
        (verbs, nouns), scores = compute_action_scores(
            collated_results["verb"], collated_results["noun"], top_n=100
        )
        collated_results["action"] = [
            {
                action_id_from_verb_noun(verb, noun): score
                for verb, noun, score in zip(
                    segment_verbs, segment_nouns, segment_score
                )
            }
            for segment_verbs, segment_nouns, segment_score in zip(verbs, nouns, scores)
        ]

    return collated_results

def scores_dict_to_ranks(scores_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: scores_to_ranks(scores) for key, scores in scores_dict.items()}


def scores_to_ranks(scores: Union[np.ndarray, List[Dict[int, float]]]) -> np.ndarray:
    if isinstance(scores, np.ndarray):
        return _scores_array_to_ranks(scores)
    elif isinstance(scores, list):
        return _scores_dict_to_ranks(scores)
    raise ValueError("Cannot compute ranks for type {}".format(type(scores)))

def _scores_array_to_ranks(scores: np.ndarray):
    """
    The rank vector contains classes and is indexed by the rank

    Examples:
        >>> _scores_array_to_ranks(np.array([[0.1, 0.15, 0.25,  0.3, 0.5], \
                                             [0.5, 0.3, 0.25,  0.15, 0.1], \
                                             [0.2, 0.4,  0.1,  0.25, 0.05]]))
        array([[4, 3, 2, 1, 0],
               [0, 1, 2, 3, 4],
               [1, 3, 0, 2, 4]])
    """
    if scores.ndim != 2:
        raise ValueError(
            "Expected scores to be 2 dimensional: [n_instances, n_classes]"
        )
    return scores.argsort(axis=-1)[:, ::-1]

def _scores_dict_to_ranks(scores: List[Dict[int, float]]) -> np.ndarray:
    """
    Compute ranking from class to score dictionary

    Examples:
        >>> _scores_dict_to_ranks([{0: 0.15, 10: 0.75, 5: 0.1},\
                                   {0: 0.85, 10: 0.10, 5: 0.05}])
        array([[10,  0,  5],
               [ 0, 10,  5]])
    """
    ranks = []
    for score in scores:
        class_ids = np.array(list(score.keys()))
        score_array = np.array([score[class_id] for class_id in class_ids])
        ranks.append(class_ids[np.argsort(score_array)[::-1]])
    return np.array(ranks)

def _check_label_predictions_preconditions(rankings: np.ndarray, labels: np.ndarray):
    if len(rankings) < 1:
        raise ValueError(
            f"Need at least one instance to evaluate, but input shape "
            f"was {rankings.shape}"
        )
    if not rankings.ndim == 2:
        raise ValueError(f"Rankings should be a 2D matrix but was {rankings.ndim}D")
    if not labels.ndim == 1:
        raise ValueError(f"Labels should be a 1D vector but was {labels.ndim}D")
    if not labels.shape[0] == rankings.shape[0]:
        raise ValueError(
            f"Number of labels ({labels.shape[0]}) provided does not match number of "
            f"predictions ({rankings.shape[0]})"
        )


def selected_topk_accuracy(rankings, labels, ks, selected_class):
    if selected_class is not None:
        idx = labels == selected_class
        rankings = rankings[idx]
        labels = labels[idx]
    return topk_accuracy(rankings, labels, ks)

def topk_accuracy(
    rankings: np.ndarray, labels: np.ndarray, ks: Union[Tuple[int, ...], int] = (1, 5)
) -> List[float]:
    """Computes TOP-K accuracies for different values of k
    Parameters:
    -----------
    rankings
        2D rankings array: shape = (instance_count, label_count)
    labels
        1D correct labels array: shape = (instance_count,)
    ks
        The k values in top-k, either an int or a list of ints.

    Returns:
    --------
    list of float: TOP-K accuracy for each k in ks

    Raises:
    -------
    ValueError
         If the dimensionality of the rankings or labels is incorrect, or
         if the length of rankings and labels aren't equal
    """
    if isinstance(ks, int):
        ks = (ks,)
    _check_label_predictions_preconditions(rankings, labels)

    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    accuracies = [tp[:, :k].max(1).mean() for k in ks]
    if any(np.isnan(accuracies)):
        raise ValueError(f"NaN present in accuracies {accuracies}")
    return accuracies

def mean_topk_recall(rankings, labels, k=5):
    classes = np.unique(labels)
    recalls = []
    for c in classes:
        recalls.append(selected_topk_accuracy(rankings, labels, ks=k, selected_class=c)[0])
    return np.mean(recalls)

def compute_anticipation_metrics(
        groundtruth_df: pd.DataFrame,
        scores: Dict[str, np.ndarray],
        tail_verbs: Sequence[int],
        tail_nouns: Sequence[int],
        unseen_participant_ids: Sequence[str],
):
    """
    Parameters
    ----------
    groundtruth_df
        DataFrame containing 'verb_class': int, 'noun_class': int and 'action_class': Tuple[int, int] columns.
    scores
        Dictionary containing three entries: 'verb', 'noun' and 'action' entries should map to a 2D
        np.ndarray of shape (instance_count, class_count) where each element is the predicted score
        of that class.
    tail_verbs
        The set of verb classes that are considered to be tail classes
    tail_nouns
        The set of noun classes that are considered to be tail classes
    unseen_participant_ids
        The set of participant IDs who do not have videos in the training set.

    Returns
    -------
    A dictionary containing nested metrics.

    Raises
    ------
    ValueError
        If the shapes of the score arrays are not correct, or the lengths of the groundtruth_df and the
        scores array are not equal, or if the grountruth_df doesn't have the specified columns.

    """
    for entry in "verb", "noun", "action":
        class_col = entry + "_class"
        if class_col not in groundtruth_df.columns:
            raise ValueError("Expected '{}' column in groundtruth_df".format(class_col))

    ranks = scores_dict_to_ranks(scores)
    top_k = 5

    overall_mt5r = {
        'verb' : mean_topk_recall(ranks["verb"], groundtruth_df["verb_class"].values, k = top_k),
        'noun': mean_topk_recall(ranks["noun"], groundtruth_df["noun_class"].values, k=top_k),
        'action': mean_topk_recall(ranks["action"], groundtruth_df["action_class"].values, k=top_k)
    }

    unseen_bool_idx = groundtruth_df.participant_id.isin(unseen_participant_ids)
    tail_verb_bool_idx = groundtruth_df.verb_class.isin(tail_verbs)
    tail_noun_bool_idx = groundtruth_df.noun_class.isin(tail_nouns)
    tail_action_bool_idx = tail_verb_bool_idx | tail_noun_bool_idx

    unseen_groundtruth_df = groundtruth_df[unseen_bool_idx]
    tail_verb_groundtruth_df = groundtruth_df[tail_verb_bool_idx]
    tail_noun_groundtruth_df = groundtruth_df[tail_noun_bool_idx]
    tail_action_groundtruth_df = groundtruth_df[tail_action_bool_idx]

    unseen_ranks = {
        task: task_ranks[unseen_bool_idx] for task, task_ranks in ranks.items()
    }

    tail_idxs = {
        "verb" : tail_verb_bool_idx.values,
        "noun" : tail_noun_bool_idx.values,
        "action" : tail_action_bool_idx.values
    }

    tail_ranks = {
        task: task_ranks[tail_idxs[task]] for task, task_ranks, in ranks.items()
    }


    unseen_mt5r = {
        'verb' : mean_topk_recall(unseen_ranks["verb"], unseen_groundtruth_df["verb_class"].values, k = top_k),
        'noun': mean_topk_recall(unseen_ranks["noun"], unseen_groundtruth_df["noun_class"].values, k=top_k),
        'action': mean_topk_recall(unseen_ranks["action"], unseen_groundtruth_df["action_class"].values, k=top_k)
    }

    tail_mt5r = {
        'verb': mean_topk_recall(tail_ranks["verb"], tail_verb_groundtruth_df["verb_class"].values, k=top_k),
        'noun': mean_topk_recall(tail_ranks["noun"], tail_noun_groundtruth_df["noun_class"].values, k=top_k),
        'action': mean_topk_recall(tail_ranks["action"], tail_action_groundtruth_df["action_class"].values, k=top_k)
    }

    return {
        "overall": overall_mt5r,
        "unseen": unseen_mt5r,
        "tail": tail_mt5r,
    }

parser = argparse.ArgumentParser(
    description="Evaluate EPIC-KITCHENS-100 action anticipation validation results",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "path_to_json",
    help = "Path to the json file to be evaluated"
)
parser.add_argument(
    "path_to_annotations",
    type=Path,
    help = "Path to the annotations"
)

parser.add_argument("--verb-count", default=97, type=int)
parser.add_argument("--noun-count", default=300, type=int)
parser.add_argument(
    "--action-count",
    default=100,
    type=int,
    help="Number of action predictions to consider",
)

def add_action_class_column(groundtruth_df):
    groundtruth_df["action_class"] = action_id_from_verb_noun(
        groundtruth_df["verb_class"], groundtruth_df["noun_class"]
    )
    return groundtruth_df


def is_missing_action_scores(submission):
    uids = list(submission["results"].keys())
    return "action" not in submission["results"][uids[0]]


def main(args):
    logging.basicConfig(level=logging.INFO)

    groundtruth_df_path = args.path_to_annotations / "EPIC_100_validation.pkl"

    groundtruth_df: pd.DataFrame = pd.read_pickle(groundtruth_df_path)
    if "narration_id" in groundtruth_df.columns:
        groundtruth_df.set_index("narration_id", inplace=True)
    groundtruth_df = add_action_class_column(groundtruth_df)

    tail_class_verbs = pd.read_csv(
        args.path_to_annotations / "EPIC_100_tail_verbs.csv", index_col="verb"
    ).index.values
    tail_class_nouns = pd.read_csv(
        args.path_to_annotations / "EPIC_100_tail_nouns.csv", index_col="noun"
    ).index.values
    unseen_participant_ids = pd.read_csv(
        args.path_to_annotations / "EPIC_100_unseen_participant_ids_validation.csv",
        index_col="participant_id",
    ).index.values

    submission = read_json(args.path_to_json)
    narration_ids = groundtruth_df.index.values
    validate_submission(
        submission, narration_ids, args.verb_count, args.noun_count, args.action_count, supported_challenges='action_anticipation'
    )

    mean_top5_recalls = compute_anticipation_metrics(
        groundtruth_df,
        convert_results(narration_ids, submission["results"]),
        tail_class_verbs,
        tail_class_nouns,
        unseen_participant_ids,
    )

    display_metrics = dict()
    for split in ["overall", "unseen", "tail"]:
        for task in ["verb", "noun", "action"]:
            task_mean_top5_recall = mean_top5_recalls[split][task]
            display_metrics[f"{split}_{task}_mt5r"]=task_mean_top5_recall
    display_metrics = {metric: value * 100 for metric, value in display_metrics.items()}
    for property in ["sls_pt", "sls_tl", "sls_td"]:
        display_metrics[property] = submission[property]

    print_metrics(display_metrics)

if __name__ == "__main__":
    main(parser.parse_args())
