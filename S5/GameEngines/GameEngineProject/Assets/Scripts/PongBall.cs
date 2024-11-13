using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public class PongBall : MonoBehaviour
{
    [SerializeField] private float initialSpeed;
    
    
    private Rigidbody _rigidbody;

    private void Start()
    {
        _rigidbody = GetComponent<Rigidbody>();
        var direction = Random.Range(0, 2) * 2 - 1;
        _rigidbody.velocity = new Vector3(0, 0, initialSpeed) * direction;
    }

    private void Update()
    {
    }
}
