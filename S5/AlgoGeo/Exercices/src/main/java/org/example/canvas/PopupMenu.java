package org.example.canvas;

import org.example.algorithms.ConvexHullAlgorithms;
import org.example.algorithms.Triangulation;
import org.example.graphics2d.Polygon;
import org.example.graphics2d.Vertex;

import javax.swing.*;

public class PopupMenu extends JPopupMenu {
    public PopupMenu(Canvas canvas, Polygon polygon) {
        JMenuItem mouseContained = new JMenuItem("Is Mouse Contained");
        mouseContained.addActionListener(_ -> JOptionPane.showMessageDialog(canvas, "The mouse is " + (polygon.isInside(new Vertex(canvas.getMousePosition().x, canvas.getMousePosition().y), canvas.getHeight()) ? "inside" : "outside")));
        add(mouseContained);
        JMenuItem isPolygonConvex = new JMenuItem("Is Polygon Convex");
        isPolygonConvex.addActionListener(_ -> JOptionPane.showMessageDialog(canvas, "The polygon is " + (polygon.isConvex() ? "convex" : "concave")));
        add(isPolygonConvex);
        JMenuItem item = new JMenuItem("Algorithm Extreme Edges");
        item.addActionListener(_ -> ConvexHullAlgorithms.extremeEdges(canvas.getGraphics(), canvas.getDrawables().stream().filter(Vertex.class::isInstance).map(Vertex.class::cast).toList()));
        add(item);
        JMenuItem item2 = new JMenuItem("Jarvis's March");
        item2.addActionListener(_ -> ConvexHullAlgorithms.jarvisMarch(canvas.getGraphics(), canvas.getDrawables().stream().filter(Vertex.class::isInstance).map(Vertex.class::cast).toList()));
        add(item2);
        JMenuItem item3 = new JMenuItem("Graham's Scan");
        item3.addActionListener(_ -> ConvexHullAlgorithms.grahamsScan(canvas.getGraphics(), canvas.getDrawables().stream().filter(Vertex.class::isInstance).map(Vertex.class::cast).toList()));
        add(item3);
        JMenuItem item4 = new JMenuItem("Naive triangulation");
        item4.addActionListener(_ -> new Triangulation().naiveTriangulation(canvas.getGraphics(), canvas.getDrawables().stream().filter(Vertex.class::isInstance).map(Vertex.class::cast).toList()));
        add(item4);
        JMenuItem item5 = new JMenuItem("Slow delaunay");
        item5.addActionListener(_ -> new Triangulation().slowDelaunay(canvas.getGraphics(), canvas.getDrawables().stream().filter(Vertex.class::isInstance).map(Vertex.class::cast).toList()));
        add(item5);
        JMenuItem item6 = new JMenuItem("Incremental delaunay");
        item6.addActionListener(_ -> new Triangulation().incrementalDelaunay(canvas.getGraphics(), canvas.getHeight(), canvas.getHeight(), canvas.getDrawables().stream().filter(Vertex.class::isInstance).map(Vertex.class::cast).toList()));
        add(item6);
        JMenuItem item7 = new JMenuItem("Voronoi");
        item7.addActionListener(_ -> new Triangulation().voronoi(canvas.getGraphics(), canvas.getHeight(), canvas.getHeight(), canvas.getDrawables().stream().filter(Vertex.class::isInstance).map(Vertex.class::cast).toList()));
        add(item7);
    }
}
