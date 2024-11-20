using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ScoreGoal : MonoBehaviour
{
    [SerializeField] private int playerScore = 0;
    [SerializeField] private int opponentScore = 0;

    [SerializeField] private TextMeshProUGUI scoreText;
    
    public static ScoreGoal Instance { get; private set; }

    private void Start()
    {
        Instance = this;
    }

    private void UpdateScore()
    {
        scoreText.text = $"{playerScore} : {opponentScore}";
    }

    public void IncrementPlayerScore()
    {
        playerScore++;
        UpdateScore();
    }

    public void IncrementOpponentScore()
    {
        opponentScore++;
        UpdateScore();
    }
}