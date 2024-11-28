using System;
using System.Collections.Generic;
using UnityEngine;

public class UIManager : MonoBehaviour
{
    [SerializeField] private List<GameObject> uiObjects = new List<GameObject>();
    [SerializeField] private GameObject defaultUIObject;

    private void Start()
    {
        DisableMenus();
        defaultUIObject.SetActive(true);
    }

    public GameObject GetMenuByName(string menuName)
    {
        var menu = uiObjects.Find(obj => obj.name == menuName);
        if (menu) return menu;
        throw new ArgumentException($"No UI menu found for {menuName}");
    }

    private void DisableMenus()
    {
        uiObjects.ForEach(uiObject => uiObject.SetActive(false));
    }

    public void SetActiveMenu(string menuName)
    {
        DisableMenus();
        GetMenuByName(menuName).SetActive(true);
    }
}