using System;
using UnityEngine;

public class Coin : MonoBehaviour
{
    [SerializeField] private int coinValue = 1;
    public int CoinValue => coinValue;
    
    private CoinCounter _coinCounter;

    public void SetCoin(CoinCounter coinCounter)
    {
        _coinCounter = coinCounter;
    }

    private void OnCollisionEnter(Collision other)
    {
        if (!other.gameObject.CompareTag("Player")) return;
        _coinCounter.AddValue(coinValue);
        Destroy(gameObject);
    }
}
