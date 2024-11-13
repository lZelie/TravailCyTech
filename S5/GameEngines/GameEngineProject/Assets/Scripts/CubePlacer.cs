using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class CubePlacer : MonoBehaviour
{
    [SerializeField] private GameObject cubePrefab;
    [SerializeField] private Transform cubeParent;

    private void PlaceCube()
    {
        
        if (!Input.GetMouseButtonDown(1) || Camera.main is null) return;
        
        var ray = Camera.main.ScreenPointToRay(Input.mousePosition);

        if (!Physics.Raycast(ray, out var hit, 100.0f)) return;
        var cube = Instantiate(cubePrefab, hit.point - cubePrefab.transform.localScale.x * ray.direction, Quaternion.identity, cubeParent);
    }

    private void Update()
    {
        PlaceCube();
    }
}
