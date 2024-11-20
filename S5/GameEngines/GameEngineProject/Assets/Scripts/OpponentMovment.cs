using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class OpponentMovment : MonoBehaviour
{
    [SerializeField] private Transform ballTransform;
    [SerializeField] private float moveSpeed = 5f;

    private void Update()
    {
        // Calculate the distance between the opponent paddle and the ball
        var distanceToBall = Mathf.Abs(ballTransform.position.y - transform.position.y);

        // Move the opponent paddle towards the ball's y-position
        if (distanceToBall <= 0.5f) return;
        var newPosition = transform.position;
        newPosition.y = Mathf.Lerp(transform.position.y, ballTransform.position.y, moveSpeed * Time.deltaTime);
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
