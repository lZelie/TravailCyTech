using System.Collections.Generic;
using UnityEngine;

public class WallBuilder : MonoBehaviour
{
    [SerializeField] private Transform cubes;
    [SerializeField] private Transform balls;
    [SerializeField] private KeyCode resetKey = KeyCode.R;

    private readonly List<Vector3> _cubesPositions = new();
    private readonly List<Vector3> _cubesRotations = new();

    private void Start()
    {
        foreach (Transform cube in cubes)
        {
            _cubesPositions.Add(cube.position);
            _cubesRotations.Add(cube.eulerAngles);
        }
    }

    private void ResetCubes()
    {
        for (var i = 0; i < cubes.childCount; i++)
        {
            if (i < _cubesPositions.Count && i < _cubesRotations.Count)
            {
                var cubeTransform = cubes.GetChild(i);
                cubeTransform.position = _cubesPositions[i];
                cubeTransform.eulerAngles = _cubesRotations[i];
                var cubeRigidbody = cubeTransform.GetComponent<Rigidbody>();
                cubeRigidbody.velocity = Vector3.zero;
                cubeRigidbody.angularVelocity = Vector3.zero;
            }
            else
            {
                Destroy(cubes.GetChild(i).gameObject);
            }
        }
    }

    private void ResetBalls()
    {
        foreach (Transform ball in balls)
        {
            Destroy(ball.gameObject);
        }
    }

    private void Update()
    {
        if (Input.GetKeyDown(resetKey))
        {
            ResetCubes();
            ResetBalls();
        }
    }
}