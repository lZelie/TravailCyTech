using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using Random = UnityEngine.Random;

public class LevelManager : MonoBehaviour
{
    [SerializeField] private List<Level> levels = new();
    [SerializeField] private Transform playerTransform;
    [SerializeField] private Transform enemyTransform;
    [SerializeField] private Transform levelsContainer;
    [SerializeField] private CoinCounter coinCounter;
    [SerializeField] private TextMeshProUGUI timeCounter;
    [SerializeField] private TextMeshProUGUI bestTimeCounter;
    [SerializeField] private TextMeshProUGUI timeGameOver;
    [SerializeField] private UIManager uiManager;
    [SerializeField] private LevelButton levelButton;
    [SerializeField] private Transform levelButtonsContainer;
    

    private Level _currentLevel;

    private float _levelStartTime;

    public float LevelElapsedTime => Time.time - _levelStartTime;

    private List<LevelButton> _levelButtons = new();
    
    private void Start()
    {
        foreach (var level in levels)
        {
            var lvlButton = Instantiate(levelButton, levelButtonsContainer);
            lvlButton.SetLevelButton(level, this);
            _levelButtons.Add(lvlButton);
        }
    }

    private void RefreshLevelButtons()
    {
        _levelButtons.ForEach(button => button.RefreshButtonUI());
    }

    public void DestroyLevels()
    {
        for (var i = 0; i < levelsContainer.childCount; i++)
        {
            Destroy(levelsContainer.GetChild(i).gameObject);
        }
    }

    public Level GetLevelByName(string levelName)
    {
        var level = levels.Find(x => x.name == levelName);
        if (level) return level;
        throw new ArgumentException($"No level with name {levelName} exists.");
    }

    public void SetLevel(Level level)
    {
        DestroyLevels();
        var levelObject = Instantiate(level.LevelObject, levelsContainer);
        playerTransform.position = level.StartPosition;
        var coinNumber = levelObject.Coins.Sum(coin => coin.CoinValue);
        coinCounter.SetCounter(coinNumber, this);
        foreach (var o in levelObject.Coins) o.SetCoin(coinCounter);
        _levelStartTime = Time.time;
        StartCoroutine(LevelTimerCoroutine());
        _currentLevel = level;
        var lastTimer = PlayerPrefs.GetFloat($"TIMER_{_currentLevel.name}");
        bestTimeCounter.text = lastTimer > 0 ? $"{lastTimer:F1}s" : "";
        
        playerTransform.position = level.StartPosition;
        playerTransform.gameObject.SetActive(true);

        if (levelObject.EnemyStart is not null)
        {
            enemyTransform.position = levelObject.EnemyStart.transform.position;
            enemyTransform.gameObject.SetActive(true);
        }
        
        uiManager.SetActiveMenu("IngameMenu");
    }

    public void SetLevel(string levelName)
    {
        var level = GetLevelByName(levelName);
        SetLevel(level);
    }

    public void SetRandomLevel()
    {
        if (levels.Count == 0) return;
        var level = levels[Random.Range(0, levels.Count)];
        SetLevel(level);
    }

    public void LoseLevel()
    {
        DestroyLevels();
        StopCoroutine(LevelTimerCoroutine());
        
        var levelTimer = LevelElapsedTime;
        Debug.Log($"Timer level: {levelTimer}s");
        
        uiManager.SetActiveMenu("GameOver");
        timeGameOver.text = $"Timer level: {levelTimer:F1}s";
        playerTransform.gameObject.SetActive(false);
        enemyTransform.gameObject.SetActive(false);
        RefreshLevelButtons();
    }

    public void FinishLevel()
    {
        DestroyLevels();
        StopCoroutine(LevelTimerCoroutine());
        var levelTimer = LevelElapsedTime;
        Debug.Log($"Timer level: {levelTimer}");

        var lastTimer = PlayerPrefs.GetFloat($"TIMER_{_currentLevel.name}");
        if (levelTimer < lastTimer || lastTimer <= 0)
        {
            PlayerPrefs.SetFloat($"TIMER_{_currentLevel.name}", levelTimer);
        }
        uiManager.SetActiveMenu("MainMenu");
        playerTransform.gameObject.SetActive(false);
        enemyTransform.gameObject.SetActive(false);
        RefreshLevelButtons();
    }

    private IEnumerator LevelTimerCoroutine()
    {
        while (true)
        {
            timeCounter.text = $"{LevelElapsedTime:F1}s";
            yield return new WaitForSeconds(0.1f);
        }
    }
}