package com.example.tp7

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.SwipeToDismissBox
import androidx.compose.material3.SwipeToDismissBoxValue
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.material3.rememberSwipeToDismissBoxState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.tp7.ui.theme.TP7Theme

class MainActivity : ComponentActivity() {
    private val viewModel = TravelViewModel()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            var showAddTravelCard by remember { mutableStateOf(false) }
            TP7Theme {
                val state by viewModel.state.collectAsState()

                Box(modifier = Modifier.fillMaxSize()) {
                    when (state) {
                        is TravelViewModel.States.Empty -> EmptyText()
                        is TravelViewModel.States.Content -> TravelCardList(
                            trips = (state as TravelViewModel.States.Content).trips,
                            modifier = Modifier
                                .fillMaxSize()
                                .statusBarsPadding()
                                .padding(16.dp)
                        )
                    }

                    AddTravelCard(
                        onClick = { showAddTravelCard = true },
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .padding(16.dp)
                    )

                    if (showAddTravelCard) {
                        AddTravelCardInput(
                            confirm = {
                                viewModel.addTrip(
                                    Trip(
                                        name = it,
                                        description = "La ville la plus belle du monde. I am writing things because I don't know what to write. If you have suggestion I would be happy to hear it. But, anyway, how was your day?"
                                    )
                                )
                                showAddTravelCard = false
                            },
                            cancel = { showAddTravelCard = false },
                            modifier = Modifier
                        )
                    }
                }
            }
        }

        savedInstanceState?.let { bundle ->
            bundle.getParcelableArrayList<Trip>("trips")?.let { viewModel.restoreTrips(it) }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putParcelableArrayList(
            "trips",
            ArrayList((viewModel.state.value as TravelViewModel.States.Content).trips)
        )
    }

    @OptIn(ExperimentalFoundationApi::class, ExperimentalMaterial3Api::class)
    @Composable
    fun TravelCard(trip: Trip, modifier: Modifier = Modifier) {
        var expanded by remember { mutableStateOf(false) }
        var expandedHeight by remember { mutableStateOf(0.dp) }

        val height by animateDpAsState(
            if (expanded) expandedHeight else 128.dp,
            animationSpec = spring(
                dampingRatio = Spring.DampingRatioMediumBouncy,
                stiffness = Spring.StiffnessLow
            ), label = ""
        )

        val dismissState = rememberSwipeToDismissBoxState()
        val backgroundColor = when (dismissState.dismissDirection) {
            SwipeToDismissBoxValue.StartToEnd -> Color.Red
            SwipeToDismissBoxValue.EndToStart -> Color.Green
            else -> Color.White
        }
        val icon = when (dismissState.dismissDirection) {
            SwipeToDismissBoxValue.StartToEnd -> android.R.drawable.ic_delete
            SwipeToDismissBoxValue.EndToStart -> android.R.drawable.ic_input_add
            else -> android.R.drawable.ic_delete
        }
        val alignment = when (dismissState.dismissDirection) {
            SwipeToDismissBoxValue.StartToEnd -> Alignment.CenterStart
            SwipeToDismissBoxValue.EndToStart -> Alignment.CenterEnd
            else -> Alignment.CenterEnd
        }


        LaunchedEffect(dismissState.currentValue) {
            when (dismissState.currentValue) {
                SwipeToDismissBoxValue.StartToEnd -> {
                    viewModel.removeTrip(trip)
                    dismissState.reset()
                }

                SwipeToDismissBoxValue.EndToStart -> {
                    viewModel.toggleFavorite(trip)
                    dismissState.reset()
                }

                else -> {}
            }
        }

        SwipeToDismissBox(
            state = dismissState,
            backgroundContent = {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .fillMaxSize()
                        .background(backgroundColor)
                ) {
                    Icon(
                        modifier = Modifier.minimumInteractiveComponentSize().align(alignment),
                        painter = painterResource(icon),
                        contentDescription = "",
                    )
                }
            }
        ) {
            Row(
                modifier = modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color.White)
                    .padding(8.dp)
                    .combinedClickable(
                        onClick = {},
                        onLongClick = { expanded = !expanded }
                    ),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Image(
                    painter = painterResource(R.drawable.ic_launcher_foreground),
                    contentDescription = "",
                    modifier = Modifier
                        .size(100.dp)
                        .clip(RoundedCornerShape(100.dp))
                        .background(Color.Cyan)
                )
                Column(modifier = Modifier
                    .padding(8.dp)
                    .height(height)
                    .wrapContentSize(Alignment.TopStart)
                    .onGloballyPositioned {
                        if (expanded) {
                            expandedHeight = it.size.height.dp / 2

                        }
                    }) {
                    Text(
                        text = if (trip.isFavorite) "${trip.name} ♥" else trip.name,
                        modifier = Modifier.padding(8.dp),
                        fontSize = 18.sp
                    )
                    Text(
                        text = trip.description,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier
                            .padding(8.dp)
                    )
                }
            }
        }
    }

    @Composable
    fun AddTravelCard(onClick: () -> Unit, modifier: Modifier = Modifier) {
        FloatingActionButton(onClick = onClick, modifier = modifier) {
            Image(
                painter = painterResource(android.R.drawable.ic_input_add),
                contentDescription = "",
                modifier = Modifier
                    .size(20.dp)
            )
        }
    }

    @Composable
    fun AddTravelCardInput(
        confirm: (text: String) -> Unit,
        cancel: () -> Unit,
        modifier: Modifier = Modifier
    ) {
        var text by remember { mutableStateOf("") }

        Box(
            modifier = modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.5f))
                .clickable(onClick = cancel)
        ) {
            Column(
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(16.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color.White)
                    .clickable {},
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                TextField(value = text, onValueChange = {
                    text = it
                }, label = { Text("Title") })
                Button(onClick = { confirm(text) }) {
                    Text(text = "Add")
                }
            }
        }
    }

    @Composable
    fun TravelCardList(trips: List<Trip>, modifier: Modifier = Modifier) {
        Column(modifier = modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            trips.forEach { trip ->
                TravelCard(trip)
            }
        }
    }

    @Composable
    fun EmptyText() {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(text = "Add a trip")
        }
    }

    @Composable
    @Preview
    fun AddTravelCardInputPreview() {
        AddTravelCardInput(confirm = {}, cancel = {})
    }

    @Preview
    @Composable
    fun AddTravelCardPreview() {
        AddTravelCard(onClick = {})
    }

    @Preview
    @Composable
    fun TravelCardPreview() {
        TravelCard(
            Trip(
                name = "Paris",
                description = "La ville la plus belle du monde. I am writing things because I don't know what to write. If you have suggestion I would be happy to hear it. But, anyway, how was your day?",
                true
            )
        )
    }
}