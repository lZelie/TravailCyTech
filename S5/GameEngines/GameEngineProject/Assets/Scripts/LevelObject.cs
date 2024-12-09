using System.Collections.Generic;
using JetBrains.Annotations;
using UnityEngine;

public class LevelObject : MonoBehaviour
{
    [SerializeField] private List<Coin> coins;
    public List<Coin> Coins => coins;

    [SerializeField] [CanBeNull] private Transform enemyStart;

    [CanBeNull] public Transform EnemyStart => enemyStart;
}
