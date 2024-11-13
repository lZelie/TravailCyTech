using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball : MonoBehaviour
{
    [SerializeField] private float altitude = -5.0f;

    private void Update()
    {
        if (transform.position.y < altitude)
        {
            Destroy(gameObject);
        }
    }
}