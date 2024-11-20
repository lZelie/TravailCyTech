using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private Transform topBound;
    [SerializeField] private Transform bottomBound;

    private void Update()
    {
        var verticalInput = Input.GetAxis("Vertical");

        var newPosition = transform.position;
        newPosition.y += verticalInput * moveSpeed * Time.deltaTime;

        newPosition.y = Mathf.Clamp(newPosition.y,
            bottomBound.position.y + bottomBound.localScale.y / 2 + transform.localScale.y / 2,
            topBound.position.y - topBound.localScale.y / 2 - transform.localScale.y / 2);

        transform.position = newPosition;
    }

    private void OnCollisionEnter(Collision other)
    {
        if (!other.gameObject.CompareTag("Ball")) return;
        var contact = other.contacts[0];
        var vector = contact.point - transform.position;
        other.rigidbody.AddForce(vector.normalized * 1.5f, ForceMode.VelocityChange);
    }
}