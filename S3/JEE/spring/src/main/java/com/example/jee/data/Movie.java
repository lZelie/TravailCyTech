package com.example.jee.data;

import jakarta.persistence.*;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Generated;
import lombok.Getter;
import lombok.Setter;

import java.util.Set;

@Entity
@Setter
@Getter
public class Movie {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private int id;

    @Column
    private String title;

    @Column
    @Min(0)
    @Max(9999)
    private int productionYear;

    @Column
    private String genre;

    @ManyToMany(targetEntity = Artist.class)
    private Set<Artist> artists;

    @ManyToMany(targetEntity = Artist.class)
    private Set<Artist> directors;


    @OneToOne(targetEntity = Country.class)
    private Country country;
}
