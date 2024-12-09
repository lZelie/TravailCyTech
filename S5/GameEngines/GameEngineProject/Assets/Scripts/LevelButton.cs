using TMPro;
using UnityEngine;
using UnityEngine.UIElements;
using Button = UnityEngine.UI.Button;

public class LevelButton : MonoBehaviour
{
    [SerializeField] private Button button;
    [SerializeField] private TextMeshProUGUI levelNameText;
    [SerializeField] private TextMeshProUGUI bestTimeText;

    private LevelManager _levelManager;
    private Level _level;
    
    public void SetLevelButton(Level level, LevelManager levelManager)
    {
        _levelManager = levelManager;
        _level = level;
        button.onClick.AddListener(OnButtonClicked);
        RefreshButtonUI();
    }

    public void RefreshButtonUI()
    {
        levelNameText.text = _level.name;
        var lastTimer = PlayerPrefs.GetFloat($"TIMER_{_level.name}");
        bestTimeText.text = lastTimer > 0 ? $"{lastTimer:F1}s" : "";
    }

    public void OnButtonClicked()
    {
        _levelManager.SetLevel(_level);
    }
}
