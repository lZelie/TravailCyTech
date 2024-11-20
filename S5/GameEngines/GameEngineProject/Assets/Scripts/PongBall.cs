using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using Random = UnityEngine.Random;

public class PongBall : MonoBehaviour
{
    public const float InitialSpeed = 5;
    [SerializeField] private float impulseInterval = 2f;
    [SerializeField] private float impulseForce = .1f;

    private Rigidbody _rigidbody;
    private float _lastYPosition;
    private float _timeSinceLastImpulse;

    private void Start()
    {
        _rigidbody = GetComponent<Rigidbody>();
        var direction = Random.Range(0, 2) * 2 - 1;
        _rigidbody.velocity = new Vector3(0, 0, InitialSpeed) * direction;
        _lastYPosition = transform.position.y;
        _timeSinceLastImpulse = 0f;
    }

    private void Update()
    {
        var currentYPosition = transform.position.y;
        var yPositionDifference = Mathf.Abs(currentYPosition - _lastYPosition);

        if (yPositionDifference < 0.01f)
        {
            _timeSinceLastImpulse += Time.deltaTime;
        }
        else
        {
            _timeSinceLastImpulse = 0f;
        }

        if (_timeSinceLastImpulse >= impulseInterval)
        {
            _rigidbody.AddForce(new Vector3(0, Random.Range(-impulseForce, impulseForce), 0), ForceMode.VelocityChange);
            _timeSinceLastImpulse = 0f;
        }

        _lastYPosition = currentYPosition;
    }
}
