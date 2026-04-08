def trajectory(state, submission):
    """
    Grader for the support triage tasks.
    The Hackathon platform calls this function to evaluate the final score.
    """
    # If the user already tracked cumulative_score, we can just return it!
    # The submission might be the final observation, or state might be the environment.
    
    score = 0.01

    # Attempt to extract grader_score passed from Observation
    if hasattr(submission, 'get') and submission.get('grader_score') is not None:
        score = submission.get('grader_score')
    elif hasattr(submission, 'grader_score') and submission.grader_score is not None:
        score = submission.grader_score

    # Make strictly bounded
    return max(0.01, min(0.99, float(score)))
