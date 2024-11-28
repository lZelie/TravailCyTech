using System;
using TMPro;
using UnityEngine;

public class CoinCounter : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI coinText;
    private int _currentValue = 0;
    private int _maxValue = 1;
    private LevelManager _levelManager;

    public static CoinCounter Instance { get; private set; }

    private void Start()
    {
        Instance = this;
    }

    public void SetCounter(int max, LevelManager levelManager)
    {
        _currentValue = 0;
        _maxValue = max;
        SetCounterText();
        _levelManager = levelManager;
    }

    private void SetCounterText()
    {
        coinText.text = $"{_currentValue} / {_maxValue}";
    }

    public void AddValue(int value)
    {
        _currentValue += value;
        SetCounterText();
        if (_currentValue >= _maxValue)
        {
            _levelManager.FinishLevel();
        }
    }
}
