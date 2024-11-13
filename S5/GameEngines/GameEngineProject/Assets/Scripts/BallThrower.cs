using System;
using UnityEngine;

public class BallThrower : MonoBehaviour
{
    [SerializeField] private GameObject ballPrefab;
    [SerializeField] private float forceMultiplier = 30;
    [SerializeField] private Transform ballSpawn;
    [SerializeField] private int maxBallsInPool = 5;
    [SerializeField] private float spawnCooldown = 0.5f;

    private float _spawnTime = 0;

    private void ThrowBall()
    {
        if (Time.time < _spawnTime + spawnCooldown || !Input.GetMouseButton(0) || Camera.main is null) return;
        _spawnTime = Time.time;
        var ray = Camera.main.ScreenPointToRay(Input.mousePosition);

        var ball = CreateBall(ray);
        var force = ray.direction * forceMultiplier;
        var rigidBody = ball.GetComponent<Rigidbody>();
        rigidBody.velocity = Vector3.zero;
        rigidBody.angularVelocity = Vector3.zero;
        rigidBody.AddForce(force, ForceMode.Impulse);
    }

    private GameObject CreateBall(Ray ray)
    {
        if (ballSpawn.childCount >= maxBallsInPool)
        {
            var first = ballSpawn.GetChild(0);
            first.SetAsLastSibling();
            first.position = ray.origin;
            return first.gameObject;
        }
        else
        {
            return Instantiate(ballPrefab, ray.origin, Quaternion.identity, ballSpawn);
        }
    }

    private void Update()
    {
        ThrowBall();
    }
}