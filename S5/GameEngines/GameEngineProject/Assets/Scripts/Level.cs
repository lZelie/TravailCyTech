using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

[CreateAssetMenu(fileName = "Level", menuName = "ScriptableObjects/Level", order = 1)]
public class Level : ScriptableObject
{
    
    [SerializeField] private Vector3 startPosition;
    public Vector3 StartPosition => startPosition;
    
    [SerializeField] private LevelObject levelObject;
    public LevelObject LevelObject => levelObject;
}
