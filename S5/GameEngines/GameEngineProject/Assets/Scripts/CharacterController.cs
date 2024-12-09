using System;
using UnityEngine;

public class CharacterController : MonoBehaviour
{
    private static readonly int IsWalking = Animator.StringToHash("isWalking");
    private static readonly int Walking = Animator.StringToHash("IsWalking");
    [SerializeField] private float speed = 3.0f;
    [SerializeField] private Animator animator;
    
    private Rigidbody _rb;
    
    
    private void Start()
    {
        _rb = GetComponent<Rigidbody>();
    }

    private void MoveCharacter()
    {
        var vertical = Input.GetAxis("Vertical");
        var horizontal = Input.GetAxis("Horizontal");
        
        var currentPosition = transform.position;
        var targetPosition = currentPosition + new Vector3(horizontal, 0, vertical);
        
        animator.SetBool("IsWalking", vertical != 0 || horizontal != 0);
        transform.LookAt(targetPosition);
        transform.position = Vector3.MoveTowards(currentPosition, targetPosition, speed * Time.fixedDeltaTime);
    }

    private void FixedUpdate()
    {
        MoveCharacter();
    }
}
