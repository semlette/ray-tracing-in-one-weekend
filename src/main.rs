use std::{fs::File, rc::Rc};

use image;
use rand::Rng;

fn random_scene() -> HittableList {
    let mut world = HittableList::new();

    let ground_material: Rc<Box<dyn Material>> = Rc::new(Box::new(Lambertian {
        albedo: Colour::new_with_values(0.5, 0.5, 0.5),
    }));
    world.add(Box::new(Sphere {
        center: Point3::new_with_values(0.0, -1000.0, 0.0),
        radius: 1000.0,
        material: ground_material,
    }));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_f64();
            let center = Point3::new_with_values(
                a as f64 + 0.9 * random_f64(),
                0.2,
                b as f64 + 0.9 * random_f64(),
            );

            if (center.sub(&Point3::new_with_values(4.0, 0.2, 0.0))).len() > 0.9 {
                let material: Box<dyn Material>;
                if choose_mat < 0.8 {
                    let albedo = Colour::new_random().mul(&Colour::new_random());
                    material = Box::new(Lambertian { albedo });
                } else if choose_mat < 0.95 {
                    let albedo = Colour::new_random_range(0.5, 1.0);
                    let fuzz = random_f64_minmax(0.0, 0.5);
                    material = Box::new(Metal { albedo, fuzz });
                } else {
                    material = Box::new(Dielectric { ir: 1.5 })
                }
                world.add(Box::new(Sphere {
                    center,
                    radius: 0.2,
                    material: Rc::new(material),
                }));
            }
        }
    }

    let material_1: Rc<Box<dyn Material>> = Rc::new(Box::new(Dielectric { ir: 1.5 }));
    let sphere_1 = Box::new(Sphere {
        center: Point3::new_with_values(0.0, 1.0, 0.0),
        radius: 1.0,
        material: material_1,
    });
    world.add(sphere_1);
    let material_2: Rc<Box<dyn Material>> = Rc::new(Box::new(Lambertian {
        albedo: Colour::new_with_values(0.4, 0.2, 0.1),
    }));
    let sphere_2 = Box::new(Sphere {
        center: Point3::new_with_values(-4.0, 1.0, 0.0),
        radius: 1.0,
        material: material_2,
    });
    world.add(sphere_2);
    let material_3: Rc<Box<dyn Material>> = Rc::new(Box::new(Metal {
        albedo: Colour::new_with_values(0.7, 0.6, 0.5),
        fuzz: 0.0,
    }));
    let sphere_3 = Box::new(Sphere {
        center: Point3::new_with_values(4.0, 1.0, 0.0),
        radius: 1.0,
        material: material_3,
    });
    world.add(sphere_3);

    world
}

fn main() {
    let aspect_ratio = 16.0 / 9.0;
    let img_width: u32 = 1200;
    let img_height: u32 = ((img_width as f64) / aspect_ratio) as u32;
    let samples_per_pixel = 500;
    let max_depth = 50;

    let world = random_scene();

    let out = File::create("out.png").unwrap();
    let mut vec = Box::new(Vec::<u8>::with_capacity((img_width * img_height) as usize));

    let look_from = Point3::new_with_values(13.0, 2.0, 3.0);
    let look_at = Point3::new_with_values(0.0, 0.0, 0.0);
    let vertical_up = Vec3::new_with_values(0.0, 1.0, 0.0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let camera = Camera::new(
        &look_from,
        &look_at,
        &vertical_up,
        20.0,
        aspect_ratio,
        aperture,
        dist_to_focus,
    );

    for j in (0..(img_height)).rev() {
        println!("scanlines remaining: {}", j);
        for i in 0..img_width {
            let mut pixel_colour = Colour::new();
            for _ in 0..samples_per_pixel {
                let u = (i as f64 + random_f64()) / (img_width as f64 - 1.0);
                let v = (j as f64 + random_f64()) / (img_height as f64 - 1.0);
                let ray = camera.get_ray(u, v);
                let colour = ray_colour(&ray, &world, max_depth);
                pixel_colour = pixel_colour.add(&colour);
            }
            write_colour(&mut vec, &pixel_colour, samples_per_pixel);
        }
    }

    let encoder = image::png::PngEncoder::new(out);
    encoder
        .encode(&vec, img_width, img_height, image::ColorType::Rgb8)
        .expect("encode image");
}

#[derive(Clone, Copy)]
struct Vec3(f64, f64, f64);

impl Vec3 {
    fn new() -> Self {
        Self(0.0, 0.0, 0.0)
    }

    fn new_with_values(a: f64, b: f64, c: f64) -> Self {
        Self(a, b, c)
    }

    fn new_random() -> Self {
        Self(random_f64(), random_f64(), random_f64())
    }

    fn new_random_range(min: f64, max: f64) -> Self {
        Self(
            random_f64_minmax(min, max),
            random_f64_minmax(min, max),
            random_f64_minmax(min, max),
        )
    }

    fn is_near_zero(&self) -> bool {
        let s = 1e-8;
        self.0.abs() < s && self.1.abs() < s && self.2.abs() < s
    }

    fn add(&self, other: &Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }

    #[allow(dead_code)]
    fn add_f64(&self, add: f64) -> Self {
        Self(self.0 + add, self.1 + add, self.2 + add)
    }

    fn mul(&self, other: &Self) -> Self {
        Self(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }

    fn mul_f64(&self, float: f64) -> Self {
        Self(self.0 * float, self.1 * float, self.2 * float)
    }

    #[allow(dead_code)]
    fn div(&self, other: &Self) -> Self {
        Self(self.0 / other.0, self.1 / other.1, self.2 / other.2)
    }

    fn div_f64(&self, float: f64) -> Self {
        Self(self.0 / float, self.1 / float, self.2 / float)
    }

    fn sub(&self, other: &Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }

    #[allow(dead_code)]
    fn sub_f64(&self, sub: f64) -> Self {
        Self(self.0 - sub, self.1 - sub, self.2 - sub)
    }

    fn neg(&self) -> Self {
        Self(-self.0, -self.1, -self.2)
    }
}

impl std::fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}, {}, {})", self.0, self.1, self.2))
    }
}

type Colour = Vec3;

impl Colour {
    fn r(&self) -> f32 {
        self.0 as f32
    }

    fn g(&self) -> f32 {
        self.1 as f32
    }

    fn b(&self) -> f32 {
        self.2 as f32
    }
}

fn write_colour(out: &mut Vec<u8>, colour: &Colour, samples_per_pixel: u32) {
    let scale = 1.0 / samples_per_pixel as f32;
    let r = (colour.r() * scale as f32).sqrt();
    let g = (colour.g() * scale as f32).sqrt();
    let b = (colour.b() * scale as f32).sqrt();

    out.push((256.0 * r.clamp(0.0, 0.999)) as u8);
    out.push((256.0 * g.clamp(0.0, 0.999)) as u8);
    out.push((256.0 * b.clamp(0.0, 0.999)) as u8);
}

type Point3 = Vec3;

impl Point3 {
    fn x(&self) -> f64 {
        self.0
    }

    fn y(&self) -> f64 {
        self.1
    }

    #[allow(dead_code)]
    fn z(&self) -> f64 {
        self.2
    }

    fn len(&self) -> f64 {
        self.len_squared().sqrt()
    }

    fn len_squared(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
    }

    fn dot(&self, other: &Self) -> f64 {
        other.0 * self.0 + other.1 * self.1 + other.2 * self.2
    }

    fn cross(&self, other: &Self) -> Self {
        Self(
            self.1 * other.2 - self.2 * other.1,
            self.2 * other.0 - self.0 * other.2,
            self.0 * other.1 - self.1 * other.0,
        )
    }

    fn unit_vector(&self) -> Self {
        self.div_f64(self.len())
    }
}

struct Ray {
    origin: Point3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Point3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    fn at(&self, t: f64) -> Point3 {
        self.origin.add(&self.direction.mul_f64(t))
    }
}

fn ray_colour(ray: &Ray, world: &dyn Hittable, depth: i32) -> Colour {
    let mut hit_record = HitRecord::new();

    if depth <= 0 {
        return Colour::new();
    }

    if world.hit(ray, 0.001, std::f64::INFINITY, &mut hit_record) {
        let mut scattered = Ray::new(Vec3::new(), Vec3::new());
        let mut attenuation = Vec3::new();

        let material = match hit_record.material {
            Some(ref material) => material.clone(),
            None => return Colour::new(),
        };

        if material.scatter(ray, &hit_record, &mut attenuation, &mut scattered) {
            return attenuation.mul(&ray_colour(&scattered, world, depth - 1));
        }
    }
    let unit_direction = ray.direction.unit_vector();
    let t = 0.5 * (unit_direction.y() + 1.0);
    return Colour::new_with_values(1.0, 1.0, 1.0)
        .mul_f64(1.0 - t)
        .add(&Colour::new_with_values(0.5, 0.7, 1.0).mul_f64(t));
}

#[derive(Clone)]
struct HitRecord {
    point: Point3,
    normal: Vec3,
    material: Option<Rc<Box<dyn Material>>>,
    t: f64,
    front_face: bool,
}

impl HitRecord {
    fn new() -> Self {
        Self {
            point: Point3::new(),
            normal: Vec3::new(),
            material: None,
            t: 0.0,
            front_face: false,
        }
    }

    fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vec3) {
        self.front_face = ray.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            outward_normal.neg()
        };
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, record: &mut HitRecord) -> bool;
}

struct Sphere {
    center: Point3,
    radius: f64,
    material: Rc<Box<dyn Material>>,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, record: &mut HitRecord) -> bool {
        let oc = ray.origin.sub(&self.center);
        let a = ray.direction.len_squared();
        let half_b = oc.dot(&ray.direction);
        let c = oc.len_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return false;
        }

        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return false;
            }
        }

        record.t = root;
        record.point = ray.at(record.t);
        let outward_normal = record.point.sub(&self.center).div_f64(self.radius);
        record.set_face_normal(ray, &outward_normal);
        record.material = Some(self.material.clone());
        return true;
    }
}

struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    fn add(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, record: &mut HitRecord) -> bool {
        let mut temp_record = record.clone();
        let mut hit_anything = false;
        let mut closest_so_far = t_max;

        for obj in &self.objects {
            if obj.hit(ray, t_min, closest_so_far, &mut temp_record) {
                hit_anything = true;
                closest_so_far = temp_record.t;
                *record = temp_record.clone();
            }
        }

        hit_anything
    }
}

fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

fn random_f64() -> f64 {
    rand::thread_rng().gen()
}

fn random_f64_minmax(min: f64, max: f64) -> f64 {
    rand::thread_rng().gen_range(min..max)
}

fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = Vec3::new_random_range(-1.0, 1.0);
        if p.len_squared() >= 1.0 {
            continue;
        }
        return p;
    }
}

fn random_unit_vector() -> Vec3 {
    random_in_unit_sphere().unit_vector()
}

fn random_in_unit_disk() -> Vec3 {
    loop {
        let p = Vec3::new_with_values(
            random_f64_minmax(-1.0, 1.0),
            random_f64_minmax(-1.0, 1.0),
            0.0,
        );
        if p.len_squared() > 1.0 {
            continue;
        }
        return p;
    }
}

fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    return v.sub(&n.mul_f64(v.dot(n) * 2.0));
}

fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = uv.neg().dot(n).min(1.0);
    let r_out_perp = uv.add(&n.mul_f64(cos_theta)).mul_f64(etai_over_etat);
    let r_out_parallel = n.mul_f64(-((1.0 - r_out_perp.len_squared()).abs().sqrt()));
    return r_out_perp.add(&r_out_parallel);
}

struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
}

impl Camera {
    fn new(
        look_from: &Point3,
        look_at: &Point3,
        vertical_up: &Vec3,
        vertical_fov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = degrees_to_radians(vertical_fov);
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = look_from.sub(look_at).unit_vector();
        let u = vertical_up.cross(&w).unit_vector();
        let v = w.cross(&u);

        let origin = *look_from;
        let horizontal = u.mul_f64(viewport_width).mul_f64(focus_dist);
        let vertical = v.mul_f64(viewport_height).mul_f64(focus_dist);
        let lower_left_corner = origin
            .sub(&horizontal.div_f64(2.0))
            .sub(&vertical.div_f64(2.0))
            .sub(&w.mul_f64(focus_dist));

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius: aperture / 2.0,
        }
    }

    fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = random_in_unit_disk().mul_f64(self.lens_radius);
        let offset = self.u.mul_f64(rd.x()).add(&self.v.mul_f64(rd.y()));

        Ray::new(
            self.origin.add(&offset),
            self.lower_left_corner
                .add(&self.horizontal.mul_f64(s))
                .add(&self.vertical.mul_f64(t))
                .sub(&self.origin)
                .sub(&offset),
        )
    }
}

trait Material {
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        attenuation: &mut Colour,
        scattered: &mut Ray,
    ) -> bool;
}

struct Lambertian {
    albedo: Colour,
}

impl Material for Lambertian {
    fn scatter(
        &self,
        _ray: &Ray,
        hit_record: &HitRecord,
        attenuation: &mut Colour,
        scattered: &mut Ray,
    ) -> bool {
        let mut scatter_direction = hit_record.normal.sub(&random_unit_vector());
        if scatter_direction.is_near_zero() {
            scatter_direction = hit_record.normal;
        }
        *scattered = Ray::new(hit_record.point, scatter_direction);
        *attenuation = self.albedo;

        true
    }
}

struct Metal {
    albedo: Colour,
    fuzz: f64,
}

impl Material for Metal {
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        attenuation: &mut Colour,
        scattered: &mut Ray,
    ) -> bool {
        let reflected = reflect(&ray.direction.unit_vector(), &hit_record.normal);
        *scattered = Ray::new(
            hit_record.point,
            reflected.add(&random_in_unit_sphere().mul_f64(self.fuzz)),
        );
        *attenuation = self.albedo;
        scattered.direction.dot(&hit_record.normal) > 0.0
    }
}

struct Dielectric {
    ir: f64,
}

impl Material for Dielectric {
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        attenuation: &mut Colour,
        scattered: &mut Ray,
    ) -> bool {
        *attenuation = Colour::new_with_values(1.0, 1.0, 1.0);
        let refraction_ratio = if hit_record.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };

        let unit_direction = ray.direction.unit_vector();
        let cos_theta = unit_direction.neg().dot(&hit_record.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction;

        if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > random_f64() {
            direction = reflect(&unit_direction, &hit_record.normal);
        } else {
            direction = refract(&unit_direction, &hit_record.normal, refraction_ratio);
        }

        *scattered = Ray::new(hit_record.point, direction);

        true
    }
}

impl Dielectric {
    fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
        let mut r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
        r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)
    }
}
