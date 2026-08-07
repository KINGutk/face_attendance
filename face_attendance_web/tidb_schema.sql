-- =========================================================
-- Face Attendance DB — TiDB Cloud Compatible SQL Schema
-- Compatible with TiDB / MySQL 8.0+
-- =========================================================

CREATE DATABASE IF NOT EXISTS `face_attendance_db` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `face_attendance_db`;

-- Disable foreign key checks for clean drop order
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS `leaves`;
DROP TABLE IF EXISTS `attendance`;
DROP TABLE IF EXISTS `detection_logs`;
DROP TABLE IF EXISTS `classes`;
DROP TABLE IF EXISTS `students`;
DROP TABLE IF EXISTS `professors`;
DROP TABLE IF EXISTS `admins`;

SET FOREIGN_KEY_CHECKS = 1;

-- --------------------------------------------------------
-- Table Structure: admins
-- --------------------------------------------------------
CREATE TABLE `admins` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(100) NOT NULL,
  `password_hash` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_admin_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Default admin account (Username: admin, Password: adminpassword / hashed)
INSERT INTO `admins` (`id`, `username`, `password_hash`) VALUES
(1, 'admin', '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9');

-- --------------------------------------------------------
-- Table Structure: professors
-- --------------------------------------------------------
CREATE TABLE `professors` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(100) DEFAULT NULL,
  `email` VARCHAR(100) DEFAULT NULL,
  `password` VARCHAR(255) DEFAULT NULL,
  `status` VARCHAR(20) NOT NULL DEFAULT 'pending',
  PRIMARY KEY (`id`),
  KEY `idx_prof_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `professors` (`id`, `name`, `email`, `password`, `status`) VALUES
(8, 'prof yonus saib', 'ynunuskhattak@gmail.com', 'scrypt:32768:8:1$sHmu9SsAqm2nuNN3$8da66e8fe060d0a59e9ac9877643529f893ccb3f82e40180f69ead68c0a445d68d698089cb7e28e12bfb714a48f2d427d114c8a890431477b99af083e996cf69', 'approved'),
(9, 'prof irfan', 'irfan@gmail.com', 'scrypt:32768:8:1$7uejoBs1stQ0uRxv$50b37596d515be6daee93b484a0f040c2786501d005d4726fdfe3949f7bb395e7fb0188119f94cc0a2af3dec1806a8a6f40fafdde0f548c2723417fb30540109', 'approved');

-- --------------------------------------------------------
-- Table Structure: students
-- --------------------------------------------------------
CREATE TABLE `students` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(100) DEFAULT NULL,
  `roll_no` VARCHAR(50) DEFAULT NULL,
  `semester` VARCHAR(20) NOT NULL DEFAULT '1st Semester',
  `email` VARCHAR(100) DEFAULT NULL,
  `image_path` VARCHAR(255) DEFAULT NULL,
  `password` VARCHAR(255) DEFAULT NULL,
  `status` ENUM('pending','approved','rejected') DEFAULT 'approved',
  `registration_date` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_student_roll_no` (`roll_no`),
  KEY `idx_student_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `students` (`id`, `name`, `roll_no`, `semester`, `email`, `image_path`, `password`, `status`, `registration_date`) VALUES
(73, 'Naveed khan', '25476', '5th Semester', 'naveedkhan20242@gmail.com', 'faces/25476_Naveed_khan/profile.jpg', 'scrypt:32768:8:1$gyVos5CptNZvAKsn$081605ec3a868376ae269219372d09168b5d2522cda25fd5b77125c226e4938ff41b6980b9a482a2af7c3eadaa70b8c47707e1cb919e06e0ec110fc2e0bbd8fc', 'approved', '2026-01-05 14:10:00'),
(74, 'zaman jan', '0987678', '2nd Semester', 'ynunuskhattak@gmail.com', 'faces/0987678_zaman_jan/front.jpg', 'scrypt:32768:8:1$d5XkhwRJJlyh9hP3$e85977833e361c6526ae488b4343e8b330b8b2d5b74656c48e5ae04f7a4336a9477cfd4b12478a694c6d1dde40a4f71a7c1d6c88be7d5a060621c644e1df6fc0', 'approved', '2026-01-07 08:47:35'),
(75, 'prof yonus saib', '17201', '2nd Semester', 'profaftab@gmail.com', 'faces/17201_prof_yonus_saib/front.jpg', 'scrypt:32768:8:1$d8HoAkYI7TUVJoqk$b43466c266f37a5b8fe1b1c3ddd120c030371ad632643b1a3125ab6c752ababd936e199d4fd5c7d9a690ab3fdb8f48efba73ff8a433d0cf0e1d1866a0bf64541', 'approved', '2026-01-07 09:06:53'),
(76, 'kaliwal', '123432', '2nd Semester', 'kaliwal@gmail.com', 'faces/123432_kaliwal/front.jpg', 'scrypt:32768:8:1$khNCSVY31OP75fF7$a16f84296d657aaa260d475b65cb40a4d69270ed0da76b48bfb8471d16234716a66dafdccc79b46b2ea1ac00e7238be7a2d965d0846bc7c60bd6f9fc177b7734', 'approved', '2026-01-07 09:13:29'),
(77, 'Abdul kabeer', '233706', '5th Semester', 'kabeer50005@gmail.com', 'faces/233706_Abdul_kabeer_/front.jpg', 'scrypt:32768:8:1$pexyxfmZ6nSNb2aj$d0414aee3f9d3e71242d21ecadea84c0531ee2d4178d1ef24381ba87f163ace71d8855b502d57a2b52301d25d9b9ba895cb3a9c1562543feb454848def00b297', 'approved', '2026-01-07 10:05:24'),
(78, 'Kami', '3737473', '7th Semester', 'alam@gmail.com', 'faces/3737473_Kami/front.jpg', 'scrypt:32768:8:1$syaNmYRgzrReTNkr$4c22777660a21025a1f4988fff9cbe437a3c5b7908c79210882cfd05eaf3b1f46b8556b4ec67029f8ccd048646992ffa2d7823c45888f7c10ab91d964d41e096', 'approved', '2026-01-08 06:18:07');

-- --------------------------------------------------------
-- Table Structure: classes
-- --------------------------------------------------------
CREATE TABLE `classes` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `subject_name` VARCHAR(100) DEFAULT NULL,
  `day_of_week` VARCHAR(10) DEFAULT NULL,
  `start_time` TIME DEFAULT NULL,
  `end_time` TIME DEFAULT NULL,
  `professor_id` INT DEFAULT NULL,
  `semester` VARCHAR(50) NOT NULL DEFAULT '1st Semester',
  PRIMARY KEY (`id`),
  KEY `fk_class_prof` (`professor_id`),
  CONSTRAINT `fk_class_prof` FOREIGN KEY (`professor_id`) REFERENCES `professors` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `classes` (`id`, `subject_name`, `day_of_week`, `start_time`, `end_time`, `professor_id`, `semester`) VALUES
(62, 'statestic', 'Thursday', '08:50:00', '09:00:00', 8, '5th Semester'),
(63, 'bio', 'Thursday', '09:04:00', '09:40:00', 8, '5th Semester');

-- --------------------------------------------------------
-- Table Structure: attendance
-- --------------------------------------------------------
CREATE TABLE `attendance` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `student_id` INT DEFAULT NULL,
  `date` DATE DEFAULT NULL,
  `time` TIME DEFAULT NULL,
  `status` VARCHAR(10) DEFAULT NULL,
  `class_id` INT DEFAULT NULL,
  `method` VARCHAR(20) DEFAULT 'face',
  PRIMARY KEY (`id`),
  KEY `idx_attendance_class_date` (`class_id`,`date`),
  KEY `idx_attendance_student_class_date` (`student_id`,`class_id`,`date`),
  CONSTRAINT `attendance_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `students` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `attendance_ibfk_2` FOREIGN KEY (`class_id`) REFERENCES `classes` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `attendance` (`id`, `student_id`, `date`, `time`, `status`, `class_id`, `method`) VALUES
(193, 77, '2026-01-08', '08:51:41', 'Present', 62, 'face'),
(194, 73, '2026-01-08', '09:00:00', 'Absent', 62, 'auto'),
(195, 77, '2026-01-08', '09:04:52', 'Present', 63, 'face'),
(196, 73, '2026-01-08', '09:14:00', 'Absent', 63, 'auto'),
(197, 73, '2026-02-21', '08:50:00', 'Present', 62, 'manual'),
(198, 77, '2026-02-21', '08:50:00', 'Present', 62, 'manual');

-- --------------------------------------------------------
-- Table Structure: detection_logs
-- --------------------------------------------------------
CREATE TABLE `detection_logs` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) DEFAULT NULL,
  `roll_no` VARCHAR(100) DEFAULT NULL,
  `subject` VARCHAR(255) DEFAULT NULL,
  `status` VARCHAR(50) DEFAULT NULL,
  `timestamp` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------
-- Table Structure: leaves
-- --------------------------------------------------------
CREATE TABLE `leaves` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `student_id` INT NOT NULL,
  `start_date` DATE NOT NULL,
  `end_date` DATE NOT NULL,
  `application_text` TEXT DEFAULT NULL,
  `status` ENUM('Pending','Approved','Rejected') DEFAULT 'Pending',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `subject_name` VARCHAR(100) DEFAULT NULL,
  `application_purpose` VARCHAR(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_leaves_student_status` (`student_id`,`status`),
  CONSTRAINT `leaves_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `students` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO `leaves` (`id`, `student_id`, `start_date`, `end_date`, `application_text`, `status`, `created_at`, `subject_name`, `application_purpose`) VALUES
(47, 77, '2026-02-18', '2026-02-19', 'im sick for today', 'Pending', '2026-02-18 14:11:27', 'statestic', 'Sick');
