package com.example.tp7

import android.os.Parcelable
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.parcelize.Parcelize

@Parcelize
data class Trip(val name: String, val description: String, var isFavorite: Boolean = false): Parcelable

class TravelViewModel: ViewModel() {
    private val _state = MutableStateFlow<States>(States.Empty)
    val state = _state.asStateFlow()

    private val _trips = mutableListOf<Trip>()

    fun addTrip(trip: Trip) {
        _trips.add(trip)
        _state.value = States.Content(_trips.toList())
    }

    fun removeTrip(trip: Trip) {
        _trips.remove(trip)
        if (_trips.isEmpty()) {
            _state.value = States.Empty
        } else {
            _state.value = States.Content(_trips.toList())
        }
    }

    fun toggleFavorite(trip: Trip) {
        val fav = trip.isFavorite
        val prevFavorite = _trips.find { it.isFavorite }
        if (prevFavorite != null) {
            prevFavorite.isFavorite = false
        }
        _state.value = States.Content(_trips.toList())
        trip.isFavorite = !fav
    }

    fun restoreTrips(trips: List<Trip>) {
        _trips.addAll(trips)
        _state.value = States.Content(_trips.toList())
    }

    sealed class States {
        data object Empty : States()
        data class Content(val trips: List<Trip>) : States()
    }

}