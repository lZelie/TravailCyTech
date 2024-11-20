using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public class GoalDetector : MonoBehaviour
{
    private enum GoalType
    {
        Player,
        Opponent,
    }

    [SerializeField] private GoalType goalType;
    
    
    private void OnCollisionEnter(Collision other)
    {
        if (!other.gameObject.CompareTag("Ball")) return;
        switch (goalType)
        {
            case GoalType.Opponent: ScoreGoal.Instance.IncrementPlayerScore();
                break;
            case GoalType.Player: ScoreGoal.Instance.IncrementOpponentScore();
                break;
            default:
                throw new ArgumentOutOfRangeException();
        }
        
        var direction = Random.Range(0, 2) * 2 - 1;
        other.rigidbody.velocity = new Vector3(0, 0, PongBall.InitialSpeed) * direction;
        other.transform.position = Vector3.zero;
    }
}
