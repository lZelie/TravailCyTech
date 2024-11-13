using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    [SerializeField] float moveSpeed = 5f;

    private Rigidbody _rigidbody;

    private void Start()
    {
        _rigidbody = GetComponent<Rigidbody>();
    }

    private void Update()
    {
        var vertical = Input.GetAxis("Vertical");
        var clampedVertical = Mathf.Clamp(vertical, -1, 1);
        transform.Translate(0, clampedVertical * moveSpeed * Time.deltaTime, 0);
    }
    
    private void OnCollisionEnter(Collision other)
    {
        if (!other.gameObject.CompareTag("Ball")) return;
        var contact = other.contacts.First();
        var vector = contact.point - transform.position;
        other.rigidbody.AddForce(vector.normalized * 10, ForceMode.VelocityChange);
    }
}