using System.Collections.Generic;
using UnityEngine;

public class LevelObject : MonoBehaviour
{
    [SerializeField] private List<Coin> coins;
    public List<Coin> Coins => coins;
}
