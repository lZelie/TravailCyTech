using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyController : MonoBehaviour
{
    private static readonly int IsWalking = Animator.StringToHash("isWalking");
    private static readonly int Walking = Animator.StringToHash("IsWalking");
    [SerializeField] private float speed = 3.0f;
    [SerializeField] private Animator animator;
    [SerializeField] private Transform target;
    [SerializeField] private LevelManager levelManager;
    
    private Rigidbody _rb;
    
    
    private void Start()
    {
        _rb = GetComponent<Rigidbody>();
    }

    private void MoveCharacter(Vector3 currentPosition, Vector3 targetPosition)
    {

        if (Vector3.Distance(currentPosition, targetPosition) <= 5.0f && Vector3.Distance(currentPosition, targetPosition) > 0.5f)
        {
            animator.SetBool("IsWalking", true);
            transform.LookAt(targetPosition);
            transform.position = Vector3.MoveTowards(currentPosition, targetPosition, speed * Time.fixedDeltaTime);
        }
        else
        {
            animator.SetBool("IsWalking", false);
        }
    }

    private void DetectTarget(Vector3 currentPosition, Vector3 targetPosition)
    {
        var dist = Vector3.Distance(currentPosition, targetPosition);
        if (dist > 2.0f) return;
        
        var ray = new Ray(currentPosition, targetPosition - transform.position);
        Physics.Raycast(ray, out var hit, dist + 1.0f);
        Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * hit.distance, Color.yellow); 
        if (hit.collider is null || !hit.collider.CompareTag("Player")) return;
        
        levelManager.LoseLevel();
    }

    private void FixedUpdate()
    {
        var currentPosition = transform.position;
        var targetPosition = target.position;
        
        MoveCharacter(currentPosition, targetPosition);
        DetectTarget(currentPosition, targetPosition);
        
        
    }
}
