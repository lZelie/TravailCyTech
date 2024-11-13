using System.Collections;
using UnityEngine;

public class Rotation : MonoBehaviour
{
    [SerializeField] private Vector3 rotationAxis;
    [SerializeField, Range(-10.0f, 10.0f)] private float angle = 1.0f;
    [SerializeField] private float startDelay = 1.0f;
    [SerializeField] private float duration = 3.0f;

    private void Start()
    {
        StartCoroutine(Rotate());
    }

    private IEnumerator Rotate()
    {
        yield return new WaitForSeconds(startDelay);

        while (Time.time < startDelay + duration)
        {
            transform.Rotate(rotationAxis, angle);
            yield return new WaitForFixedUpdate();
        }
    }
}
